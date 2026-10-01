import { PATHOLOGY_BUCKET, supabase, supabaseConfigured } from "./supabase";

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
  const { data, error } = await client
    .from("analyses")
    .select("id, kind, result, created_at")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });
  if (error) throw error;

  return data.map((analysis) => ({
    id: analysis.id,
    type: analysis.kind === "risk" ? "risk" : "image",
    result: analysis.kind === "risk"
      ? analysis.result.Prediction || "Risk assessment"
      : analysis.result.Class || analysis.result.Prediction || "Histopathology analysis",
    confidence: Number(analysis.result["Confidence (%)"] || 0),
    unknown: analysis.result.Class === "unknown",
    date: analysis.created_at,
  }));
}

export async function saveAnalysis({ type, result, patientAnswers, imageFile }) {
  const client = requireSupabase();
  const user = await getCurrentUser(client);
  const kind = type === "risk" ? "risk" : "histopathology";
  const storedResult = { ...result };
  delete storedResult.Filename;
  const { data: analysis, error } = await client
    .from("analyses")
    .insert({
      user_id: user.id,
      kind,
      result: storedResult,
      patient_answers: patientAnswers || null,
    })
    .select("id, created_at")
    .single();
  if (error) throw error;

  if (imageFile) {
    const extension = imageFile.type === "image/jpeg" ? "jpg" : imageFile.type.split("/")[1];
    const imagePath = `${user.id}/${analysis.id}/${crypto.randomUUID()}.${extension}`;
    const { error: uploadError } = await client.storage
      .from(PATHOLOGY_BUCKET)
      .upload(imagePath, imageFile, { contentType: imageFile.type, upsert: false });
    if (uploadError) {
      await client.from("analyses").delete().eq("id", analysis.id).eq("user_id", user.id);
      throw uploadError;
    }

    const { error: updateError } = await client
      .from("analyses")
      .update({ image_path: imagePath })
      .eq("id", analysis.id)
      .eq("user_id", user.id);
    if (updateError) {
      await client.storage.from(PATHOLOGY_BUCKET).remove([imagePath]);
      await client.from("analyses").delete().eq("id", analysis.id).eq("user_id", user.id);
      throw updateError;
    }
  }

  return { id: analysis.id, date: analysis.created_at };
}

export async function saveChatExchange(analysisId, userMessage, assistantMessage) {
  const client = requireSupabase();
  const user = await getCurrentUser(client);
  const { error } = await client.from("chat_messages").insert([
    { analysis_id: analysisId, user_id: user.id, role: "user", content: userMessage },
    { analysis_id: analysisId, user_id: user.id, role: "assistant", content: assistantMessage },
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

  const imagePaths = analyses.map((analysis) => analysis.image_path).filter(Boolean);
  if (imagePaths.length) {
    const { error: storageError } = await client.storage.from(PATHOLOGY_BUCKET).remove(imagePaths);
    if (storageError) throw storageError;
  }

  const { error } = await client.from("analyses").delete().eq("user_id", user.id);
  if (error) throw error;
}