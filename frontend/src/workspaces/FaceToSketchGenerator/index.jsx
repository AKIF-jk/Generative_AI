import React, { useCallback, useEffect, useRef, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function FaceToSketchGenerator() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [sourceUrl, setSourceUrl] = useState('');
  const [sketchUrl, setSketchUrl] = useState('');
  const [style, setStyle] = useState(1);
  const [styles, setStyles] = useState([]);
  const [timing, setTiming] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API_URL}/api/face-to-sketch/styles`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data) => setStyles(data.styles || []))
      .catch(() => setStyles([1, 2, 3].map((id) => ({ id, name: `Style ${id}` }))));
  }, []);

  const selectFile = useCallback((candidate) => {
    const next = candidate?.[0];
    if (!next) return;
    if (!next.type.startsWith('image/')) return setError('Please choose a JPG, PNG, or WEBP image.');
    if (next.size > 20 * 1024 * 1024) return setError('Please choose an image smaller than 20 MB.');
    setError(''); setFile(next); setSourceUrl(URL.createObjectURL(next)); setSketchUrl(''); setTiming(null);
  }, []);

  const generate = async () => {
    if (!file) return setError('Upload a portrait before generating a sketch.');
    setBusy(true); setError(''); setSketchUrl(''); setTiming(null);
    const body = new FormData(); body.append('photo', file); body.append('style', String(style));
    try {
      const response = await fetch(`${API_URL}/api/face-to-sketch/generate`, { method: 'POST', body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail?.message || payload.detail || `Request failed (${response.status})`);
      setSketchUrl(payload.images.sketch); setTiming(payload.timing_ms);
    } catch (caught) { setError(caught.message || 'The server could not generate a sketch.'); }
    finally { setBusy(false); }
  };

  const reset = () => { setFile(null); setSourceUrl(''); setSketchUrl(''); setTiming(null); setError(''); if (inputRef.current) inputRef.current.value = ''; };
  const availableStyles = styles.length ? styles : [1, 2, 3].map((id) => ({ id, name: `Style ${id}` }));

  return <div className="sketch-app">
    <main id="top" className="sketch-main">
      <section className="hero"><p className="eyebrow">AI PORTRAIT STUDIO</p><h1>Turn a face into a<br /><em>hand-drawn sketch.</em></h1><p className="hero-copy">Upload a portrait and let our style-conditioned generator create a pencil-style sketch in seconds.</p></section>
      <section className="studio-grid">
        <article className="studio-card input-card">
          <div className="card-heading"><div><span className="step-label">01</span><h2>Upload a portrait</h2></div><span className="file-hint">JPG, PNG or WEBP · max 20 MB</span></div>
          {!sourceUrl ? <label className={`dropzone ${dragging ? 'is-dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files); }}><input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => selectFile(event.target.files)} /><span className="upload-icon">↥</span><strong>Drop your image here</strong><span>or <u>browse files</u></span></label> : <div className="preview-frame"><img src={sourceUrl} alt="Selected portrait" /><button className="image-remove" onClick={reset}>Remove image</button></div>}
          <div className="style-picker"><span>Sketch style</span><div className="style-options">{availableStyles.map((option) => <button key={option.id} className={style === option.id ? 'selected' : ''} onClick={() => setStyle(option.id)}>{option.name}</button>)}</div></div>
          <button className="generate-button" disabled={!file || busy} onClick={generate}>{busy ? <><span className="spinner" /> Creating sketch…</> : <>Generate sketch <span className="arrow">→</span></>}</button>
          {error && <div className="error-message" role="alert"><span>!</span>{error}</div>}
        </article>
        <article className="studio-card output-card"><div className="card-heading"><div><span className="step-label">02</span><h2>Your sketch</h2></div>{sketchUrl && <a className="download-link" href={sketchUrl} download="generated-sketch.png">Download <span>↓</span></a>}</div><div className={`result-frame ${sketchUrl ? 'has-result' : ''}`}>{sketchUrl ? <img src={sketchUrl} alt="Generated sketch" /> : <div className="empty-result"><span className="sparkle">✧</span><strong>Your result will appear here</strong><span>Upload a portrait to get started</span></div>}</div>{timing && <div className="timing"><span>Generation complete</span><strong>{timing.total} ms</strong></div>}</article>
      </section>
      <p className="privacy-note"><span>✦</span> Images are processed for this session and are not stored.</p>
    </main>
  </div>;
}

export default FaceToSketchGenerator;
