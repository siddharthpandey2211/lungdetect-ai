import { useState } from "react";
import { findProviders } from "../services/api";

const OWNERSHIP_LABELS = {
  government: "Government",
  private: "Private",
  charitable: "Charitable / Trust",
  unknown: "Ownership not specified",
};

function safeExternalUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : "";
  } catch {
    return "";
  }
}

export default function Providers({ standalone = false }) {
  const [pin, setPin] = useState("");
  const [radiusKm, setRadiusKm] = useState("25");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const loadProviders = async (event) => {
    event.preventDefault();
    const cleanPin = pin.trim();
    const radius = Number(radiusKm);
    if (!/^[1-9][0-9]{5}$/.test(cleanPin)) {
      setError("Enter a valid six-digit Indian PIN code.");
      return;
    }
    if (!Number.isFinite(radius) || radius <= 0 || radius > 100) {
      setError("Radius must be between 1 and 100 kilometres.");
      return;
    }

    setBusy(true);
    setError("");
    setSubmitted(true);
    try {
      setResult(await findProviders({ pin: cleanPin, radiusKm: radius }));
    } catch (searchError) {
      setResult(null);
      setError(searchError.message);
    } finally {
      setBusy(false);
    }
  };

  const providers = result?.providers || [];

  return (
    <div className={standalone ? "stack fade" : "panel"}>
      {!standalone && <h3>Find Lung Cancer Care</h3>}
      {standalone && (
        <p className="muted">
          Enter your PIN code and search radius to see nearby verified hospitals
          and their ownership as listed in the directory.
        </p>
      )}
      <form className="grid provider-search" onSubmit={loadProviders}>
        <label>
          Indian PIN code
          <input
            value={pin}
            onChange={(event) => setPin(event.target.value.replace(/\D/g, "").slice(0, 6))}
            inputMode="numeric"
            autoComplete="postal-code"
            maxLength="6"
            placeholder="Enter six-digit PIN code"
            required
          />
        </label>
        <label>
          Radius (km)
          <input
            type="number"
            min="1"
            max="100"
            value={radiusKm}
            onChange={(event) => setRadiusKm(event.target.value)}
            required
          />
        </label>
        <div className="provider-search-wide">
          <button className="btn" type="submit" disabled={busy}>
            {busy ? "Searching..." : "Find nearby hospitals"}
          </button>
        </div>
      </form>
      <p className="tiny">
        Search uses locally stored PIN coordinates and database-listed verified
        hospitals. Distance is approximate straight-line distance.
      </p>
      {error && (
        <div className="err" role="alert">
          {error}
          <button
            className="btn ghost provider-retry"
            type="button"
            onClick={loadProviders}
            disabled={busy}
          >
            Retry
          </button>
        </div>
      )}
      {busy && <div className="notice" aria-live="polite">Loading nearby hospitals...</div>}
      {submitted && result && !providers.length && (
        <div className="notice">
          No verified hospitals were found within {result.radius_km} km of PIN {result.origin.pin}.
        </div>
      )}
      {providers.map((provider) => {
        const websiteUrl = safeExternalUrl(provider.website_url || provider.website);
        const appointmentUrl = safeExternalUrl(provider.appointment_url);
        const phone = typeof provider.phone === "string"
          && /^[+()\d\s-]{7,}$/.test(provider.phone.trim())
          ? provider.phone.trim()
          : "";
        const location = [
          provider.address,
          provider.city,
          provider.state,
          provider.postal_code,
        ].filter(Boolean).join(", ");
        const hasCoordinates = Number.isFinite(Number(provider.latitude))
          && Number.isFinite(Number(provider.longitude));
        const mapUrl = hasCoordinates
          ? `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${provider.latitude},${provider.longitude}`)}`
          : location
            ? `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(location)}`
            : "";
        const ownership = OWNERSHIP_LABELS[provider.ownership_type] || OWNERSHIP_LABELS.unknown;

        return (
          <article className="provider-card" key={provider.id || `${provider.name}-${provider.postal_code}`}>
            <div className="provider-card-head">
              <h3>{provider.name}</h3>
              {provider.verified === true && (
                <span className="provider-verified-badge">Verified</span>
              )}
            </div>
            <p><b>Ownership:</b> {ownership}</p>
            {provider.specialty && (
              <p>{provider.specialty}{provider.hospital_affiliation ? ` · ${provider.hospital_affiliation}` : ""}</p>
            )}
            {location && <p>{location}</p>}
            <p><b>{provider.distance_km} km</b> from PIN {result.origin.pin}</p>
            {Array.isArray(provider.services) && provider.services.length > 0 && (
              <div className="service-tags">
                {provider.services.map((service) => (
                  <span key={service}>{service.replaceAll("_", " ")}</span>
                ))}
              </div>
            )}
            {provider.last_verified_at && (
              <p className="tiny">
                Directory verified {provider.last_verified_at}
                {provider.verification_source ? ` · Source: ${provider.verification_source}` : ""}
              </p>
            )}
            {provider.cost_information && <p>Financial information: {provider.cost_information}</p>}
            <div className="provider-actions">
              {phone && <a href={`tel:${phone}`}>Call {phone}</a>}
              {mapUrl && <a href={mapUrl} target="_blank" rel="noreferrer">View directions</a>}
              {websiteUrl && <a href={websiteUrl} target="_blank" rel="noreferrer">Official website</a>}
              {appointmentUrl && <a href={appointmentUrl} target="_blank" rel="noreferrer">Appointments</a>}
            </div>
          </article>
        );
      })}
      {standalone && (
        <p className="tiny">
          Only database records marked verified are shown. Government/private
          ownership is displayed as recorded; missing ownership is not guessed.
        </p>
      )}
    </div>
  );
}
