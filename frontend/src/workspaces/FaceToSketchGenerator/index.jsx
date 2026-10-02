import React, { useState, useRef } from 'react';

const FaceToSketchGenerator = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [style, setStyle] = useState(1);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inferenceTime, setInferenceTime] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setResultImage(null);
      setError(null);
    }
  };

  const handleGenerate = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('photo', selectedFile);
    formData.append('style', style);

    try {
      // Point this to your backend if running locally: http://localhost:8000/api/face-to-sketch
      const response = await fetch('/api/face-to-sketch', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Generation failed: ${errText}`);
      }

      const infTime = response.headers.get('X-Inference-Time-ms');
      if (infTime) setInferenceTime(infTime);

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResultImage(imageUrl);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Face-to-Sketch Generator</h2>
      
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Upload Photo</label>
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange}
            ref={fileInputRef}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Select Style</label>
          <div className="flex gap-4">
            {[1, 2, 3].map((s) => (
              <label key={s} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="style"
                  value={s}
                  checked={style === s}
                  onChange={() => setStyle(s)}
                  className="w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                />
                <span>Style {s}</span>
              </label>
            ))}
          </div>
        </div>

        <button 
          onClick={handleGenerate}
          disabled={loading || !selectedFile}
          className="bg-blue-600 text-white px-6 py-2 rounded font-medium disabled:opacity-50 hover:bg-blue-700 transition"
        >
          {loading ? 'Generating...' : 'Generate Sketch'}
        </button>

        {error && <div className="mt-4 p-3 bg-red-50 text-red-700 rounded">{error}</div>}
      </div>

      {(selectedFile || resultImage) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-white p-6 rounded-lg shadow-md">
          {selectedFile && (
            <div>
              <h3 className="font-semibold mb-2">Input Photo</h3>
              <img 
                src={URL.createObjectURL(selectedFile)} 
                alt="Input" 
                className="w-full max-w-sm rounded border"
              />
            </div>
          )}
          
          {resultImage && (
            <div>
              <h3 className="font-semibold mb-2 flex items-center justify-between">
                <span>Generated Sketch</span>
                {inferenceTime && <span className="text-sm font-normal text-gray-500">{inferenceTime}ms</span>}
              </h3>
              <img 
                src={resultImage} 
                alt="Generated Sketch" 
                className="w-full max-w-sm rounded border mb-4"
              />
              <a 
                href={resultImage} 
                download={`sketch_style${style}.png`}
                className="inline-block bg-gray-100 text-gray-700 px-4 py-2 rounded font-medium hover:bg-gray-200"
              >
                Download Result
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FaceToSketchGenerator;
