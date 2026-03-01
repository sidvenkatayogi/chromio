import { useState } from 'react'

export default function PaletteGenerator() {
  const [prompt, setPrompt] = useState('')
  const [palette, setPalette] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [generatedPrompt, setGeneratedPrompt] = useState('')

  const rgbToHex = (r, g, b) => {
    const toHex = (c) => {
      const hex = c.toString(16);
      return hex.length === 1 ? "0" + hex : hex;
    };
    return "#" + toHex(r) + toHex(g) + toHex(b);
  };

  const handleGenerate = async () => {
    if (!prompt || loading) return;
    setLoading(true);
    setError(null);

    try {
      console.log(import.meta.env.VITE_SERVER_URL)
      const response = await fetch(`${import.meta.env.VITE_SERVER_URL}/api/v1/text2palette/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_query: prompt }),
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => null);
        throw new Error(errBody?.message || `HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.palette && Array.isArray(data.palette)) {
        const hexPalette = data.palette.map(color => {
            // Handle if color is not array of 3
            if (Array.isArray(color) && color.length >= 3) {
                 return rgbToHex(color[0], color[1], color[2]);
            }
            return "#000000";
        });
        setPalette(hexPalette);
        setGeneratedPrompt(data.user_query);
      } else {
        throw new Error('Invalid response format from server');
      }

    } catch (err) {
      console.error(err);
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center max-w-4xl mx-auto p-8 text-center bg-gray-900">
      <h1 className="text-4xl font-bold mb-8 text-white">Text to Palette Generator</h1>

      {/* Input container */}
      <div className="flex justify-center gap-2 mb-8">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt (e.g. 'cyberpunk city')"
          onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
          className="w-72 px-4 py-2 border border-gray-600 bg-gray-800 text-white placeholder-gray-400 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none"
        />
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? 'Generating...' : 'Generate'}
        </button>
      </div>

      {/* Error message */}
      {error && <div className="text-red-400 mb-6">{error}</div>}

      {/* Palette display */}
      {palette.length > 0 && (
        <div className="bg-gray-800 rounded-xl shadow-lg p-8">
            <h2 className="text-xl font-semibold mb-6 text-white">Palette for "{generatedPrompt}"</h2>

            {/* Palette container */}
            <div className="flex flex-wrap justify-center gap-4 mb-8">
            {palette.map((color, index) => (
                <div
                  key={index}
                  className="w-36 h-36 rounded-xl shadow-md hover:scale-105 transition-transform flex items-center justify-center"
                  style={{ backgroundColor: color }}
                >
                  <span className="bg-gray-900/80 backdrop-blur px-3 py-2 rounded font-mono font-bold text-white">
                    {color}
                  </span>
                </div>
            ))}
            </div>

            {/* Retry container */}
            <div className="flex flex-col items-center">
                <button
                  onClick={handleGenerate}
                  disabled={loading}
                  className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors disabled:opacity-50"
                >
                    Retry Same Prompt
                </button>
                <p className="text-gray-400 text-sm mt-4">
                    Want to refine? Edit the text above and click Generate.
                </p>
            </div>
        </div>
      )}
    </div>
  )
}