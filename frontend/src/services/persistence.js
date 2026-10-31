import {
  CT_BUCKET,
  MRI_BUCKET,
  PATHOLOGY_BUCKET,
  supabase,
  supabaseConfigured,
} from "./supabase";

const SCAN_TABLES = {
  mri: { table: "mri_analyses", bucket: MRI_BUCKET },
  ct: { table: "ct_scan_analyses", bucket: CT_BUCKET },
};

function requireSupabase() {
  if (!supabaseConfigured || !supabase) {
    throw new Error("Supabase is not configured. Add its URL and anon key to frontend/.env.local.");
  }
  return supabase;
}

async function getCurrentUser(client) {
  const { data, error } = await client.auth.getUser();
  if (error) throw error;
  if (!data.user) throw new Error("Sign in before saving health data.");
  return data.user;
}

export async function loadAnalyses() {
  const client = requireSupabase();
  const user = await getCurrentUser(client);
  const [analyses, mriAnalyses, ctAnalyses] = await Promise.all([
    client.from("analyses").select("id, kind, result, created_at").eq("user_id", user.id),
    client.from("mri_analyses").select("id, result, created_at").eq("user_id", user.id),
    client.from("ct_scan_analyses").select("id, result, created_at").eq("user_id", user.id),
  ]);
  for (const response of [analyses, mriAnalyses, ctAnalyses]) {
    if (response.error) throw response.error;
  }

  const rows = [
    ...analyses.data.map((analysis) => ({
      ...analysis,
      type: analysis.kind === "risk" ? "risk" : "image",
    })),
    ...mriAnalyses.data.map((analysis) => ({ ...analysis, type: "mri" })),
    ...ctAnalyses.data.map((analysis) => ({ ...analysis, type: "ct" })),
  ];

  return rows
    .map((analysis) => ({
      id: analysis.id,
      type: analysis.type,
      result: analysis.type === "risk"
        ? analysis.result.Prediction || "Risk assessment"
        : analysis.result.Class || analysis.result.Prediction || "Image analysis",
      confidence: Number(analysis.result["Confidence (%)"] || 0),
      unknown: analysis.result.Class === "unknown",
      date: analysis.created_at,
    }))
    .sort((left, right) => new Date(right.date) - new Date(left.date));
}

export async function saveAnalysis({ type, result, patientAnswers, imageFile }) {
  const client = requireSupabase();
  const user = await getCurrentUser(client);
  const scan = SCAN_TABLES[type];
  if (imageFile && !["image/jpeg", "image/png", "image/webp"].includes(imageFile.type)) {
    throw new Error("Only JPG, PNG, or WEBP images can be saved.");
  }
  if (scan && !imageFile) {
    throw new Error("The original scan image is required to save this analysis.");
  }
  if (scan && imageFile.size > 10 * 1024 * 1024) {
    throw new Error("Scan images must be no larger than 10 MB to save to your account.");
  }

  const storedResult = { ...result };
  delete storedResult.Filename;
  const table = scan?.table || "analyses";
  const row = scan
    ? {
        user_id: user.id,
        result: storedResult,
        image_mime_type: imageFile.type,
        image_size_bytes: imageFile.size,
      }
    : {
        user_id: user.id,
        kind: type === "risk" ? "risk" : "histopathology",
        result: storedResult,
        patient_answers: patientAnswers || null,
      };
  const { data: analysis, error } = await client
    .from(table)
    .insert(row)
    .select("id, created_at")
    .single();
  if (error) throw error;

  if (imageFile) {
    const extension = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp" }[imageFile.type];
    const imagePath = `${user.id}/${analysis.id}/${crypto.randomUUID()}.${extension}`;
    const bucket = scan?.bucket || PATHOLOGY_BUCKET;
    const { error: uploadError } = await client.storage
      .from(bucket)
      .upload(imagePath, imageFile, { contentType: imageFile.type, upsert: false });
    if (uploadError) {
      const { error: cleanupError } = await client
        .from(table)
        .delete()
        .eq("id", analysis.id)
        .eq("user_id", user.id);
      if (cleanupError) {
        throw new Error(`${uploadError.message} Analysis cleanup also failed: ${cleanupError.message}`);
      }
      throw uploadError;
    }

    const { error: updateError } = await client
      .from(table)
      .update({ image_path: imagePath })
      .eq("id", analysis.id)
      .eq("user_id", user.id);
    if (updateError) {
      const { error: storageCleanupError } = await client.storage.from(bucket).remove([imagePath]);
      const { error: rowCleanupError } = await client
        .from(table)
        .delete()
        .eq("id", analysis.id)
        .eq("user_id", user.id);
      if (storageCleanupError || rowCleanupError) {
        throw new Error(
          `${updateError.message} Cleanup failed: ${[
            storageCleanupError?.message,
            rowCleanupError?.message,
          ].filter(Boolean).join("; ")}`,
        );
      }
      throw updateError;
    }
  }

  return { id: analysis.id, date: analysis.created_at };
}

export async function saveChatExchange(analysisId, userMessage, assistantMessage, contextType = "risk") {
  const client = requireSupabase();
  const user = await getCurrentUser(client);
  const analysisReference = contextType === "mri"
    ? { mri_analysis_id: analysisId }
    : contextType === "ct"
      ? { ct_scan_analysis_id: analysisId }
      : { analysis_id: analysisId };
  const { error } = await client.from("chat_messages").insert([
    { ...analysisReference, user_id: user.id, role: "user", content: userMessage },
    { ...analysisReference, user_id: user.id, role: "assistant", content: assistantMessage },
  ]);
  if (error) throw error;
}

export async function deleteMyData() {
  const client = requireSupabase();
  const user = await getCurrentUser(client);
  const { data: analyses, error: loadError } = await client
    .from("analyses")
    .select("id, image_path")
    .eq("user_id", user.id);
  if (loadError) throw loadError;

  const [mriResponse, ctResponse] = await Promise.all([
    client.from("mri_analyses").select("id, image_path").eq("user_id", user.id),
    client.from("ct_scan_analyses").select("id, image_path").eq("user_id", user.id),
  ]);
  if (mriResponse.error) throw mriResponse.error;
  if (ctResponse.error) throw ctResponse.error;

  const imageGroups = [
    [PATHOLOGY_BUCKET, analyses],
    [MRI_BUCKET, mriResponse.data],
    [CT_BUCKET, ctResponse.data],
  ];
  for (const [bucket, rows] of imageGroups) {
    const paths = rows.map((analysis) => analysis.image_path).filter(Boolean);
    if (paths.length) {
      const { error: storageError } = await client.storage.from(bucket).remove(paths);
      if (storageError) throw storageError;
    }
  }

  const deletes = await Promise.all([
    client.from("analyses").delete().eq("user_id", user.id),
    client.from("mri_analyses").delete().eq("user_id", user.id),
    client.from("ct_scan_analyses").delete().eq("user_id", user.id),
  ]);
  const deleteError = deletes.find((response) => response.error)?.error;
  if (deleteError) throw deleteError;
}