import React, { useRef, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const corruptions = [
  { value: 'clean', label: 'Clean image' },
  { value: 'salt_pepper', label: 'Salt-and-pepper noise' },
  { value: 'gaussian_blur', label: 'Gaussian blur' },
  { value: 'occlusion', label: 'Occlusion' },
];

export default function RestorationWorkspace({ task }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [corruption, setCorruption] = useState('gaussian_blur');
  const [severity, setSeverity] = useState('medium');
  const [routing, setRouting] = useState('predicted');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const isTask2 = task === 2;
  const endpoint = isTask2 ? '/api/hard-routing/restore' : '/api/universal/restore';

  const choose = (files) => {
    const next = files?.[0]; if (!next) return;
    if (!next.type.startsWith('image/')) return setError('Please select an image file.');
    setFile(next); setPreview(URL.createObjectURL(next)); setResult(null); setError('');
  };

  const restore = async () => {
    if (!file) return setError('Upload an image first.');
    setBusy(true); setError('');
    const body = new FormData(); body.append('image', file); body.append('corruption', corruption);
    if (corruption !== 'clean') body.append('severity', severity);
    if (isTask2) body.append('routing', routing);
    try {
      const response = await fetch(`${API_URL}${endpoint}`, { method: 'POST', body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail?.message || payload.detail || `Request failed (${response.status})`);
      setResult(payload);
    } catch (caught) { setError(caught.message || 'Restoration failed.'); }
    finally { setBusy(false); }
  };

  return <div className="restoration-page">
    <section className="workspace-intro"><p className="eyebrow">TASK {task} · IMAGE RESTORATION</p><h1>{isTask2 ? <>Route every corruption to<br /><em>the right specialist.</em></> : <>One model for<br /><em>every corruption.</em></>}</h1><p className="hero-copy">Upload an image, choose an evaluation corruption, and inspect the restoration quality and metrics.</p></section>
    <section className="restore-layout">
      <article className="studio-card restore-controls"><div className="card-heading"><div><span className="step-label">01</span><h2>Configure input</h2></div><span className="file-hint">128 × 128 model input</span></div>
        {!preview ? <label className="dropzone compact-drop"><input ref={inputRef} type="file" accept="image/*" onChange={(e) => choose(e.target.files)} /><span className="upload-icon">↥</span><strong>Drop an image here</strong><span>or <u>browse files</u></span></label> : <div className="preview-frame compact-preview"><img src={preview} alt="Input" /><button className="image-remove" onClick={() => { setFile(null); setPreview(''); setResult(null); }}>Remove</button></div>}
        <label className="field-label">Corruption applied before restoration<select value={corruption} onChange={(e) => setCorruption(e.target.value)}>{corruptions.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        {corruption !== 'clean' && <label className="field-label">Severity<select value={severity} onChange={(e) => setSeverity(e.target.value)}><option>low</option><option>medium</option><option>high</option></select></label>}
        {isTask2 && <label className="field-label">Routing mode<select value={routing} onChange={(e) => setRouting(e.target.value)}><option value="predicted">Predicted classifier route</option><option value="oracle">Oracle route</option></select></label>}
        <button className="generate-button" disabled={!file || busy} onClick={restore}>{busy ? <><span className="spinner" /> Restoring…</> : <>Restore image <span className="arrow">→</span></>}</button>
        {error && <div className="error-message" role="alert"><span>!</span>{error}</div>}
      </article>
      <article className="studio-card restore-output"><div className="card-heading"><div><span className="step-label">02</span><h2>Restoration result</h2></div>{result && <a className="download-link" href={result.downloads.output} download="restored.png">Download <span>↓</span></a>}</div>
        {result ? <><div className="image-trio"><figure><img src={result.images.input} alt="Corrupted input" /><figcaption>Input</figcaption></figure><figure><img src={result.images.output} alt="Restored result" /><figcaption>Restored</figcaption></figure><figure><img src={result.images.error_heatmap} alt="Error heatmap" /><figcaption>Error heatmap</figcaption></figure></div><div className="metric-row"><div><span>SSIM</span><strong>{result.metrics.ssim}</strong></div><div><span>L1 error</span><strong>{result.metrics.l1}</strong></div><div><span>PSNR</span><strong>{result.metrics.psnr} dB</strong></div><div><span>Time</span><strong>{result.timing_ms.total} ms</strong></div></div>{isTask2 && result.routing && <div className="route-note"><strong>{result.routing.selected_expert}</strong><span>{Math.round(result.routing.confidence * 100)}% classifier confidence</span></div>}</> : <div className="result-frame empty-result"><span className="sparkle">✧</span><strong>Your result will appear here</strong><span>Configure an input to get started</span></div>}
      </article>
    </section>
  </div>;
}
