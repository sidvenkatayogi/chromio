import { useState } from 'react'
import './App.css'

function App() {
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
    if (!prompt) return;
    setLoading(true);
    setError(null);
    
    // Keep previous palette until new one loads? Or clear? 
    // Clearing feels more responsive to "action started".
    // setPalette([]); 

    try {
      const response = await fetch(`${import.meta.env.VITE_SERVER_URL}/api/v1/text2palette/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_query: prompt }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
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
    <div>
      <h1>Text to Palette Generator</h1>
      
      <div className="input-container">
        <input 
          type="text" 
          value={prompt} 
          onChange={(e) => setPrompt(e.target.value)} 
          placeholder="Enter a prompt (e.g. 'cyberpunk city')"
          onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
        />
        <button onClick={handleGenerate} disabled={loading}>
          {loading ? 'Generating...' : 'Generate'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {palette.length > 0 && (
        <div className="card">
            <h2>Palette for "{generatedPrompt}"</h2>
            <div className="palette-container">
            {palette.map((color, index) => (
                <div key={index} className="color-square" style={{ backgroundColor: color }}>
                <span className="color-hex">{color}</span>
                </div>
            ))}
            </div>
            
            <div className="retry-container">
                <button onClick={handleGenerate} disabled={loading}>
                    Retry Same Prompt
                </button>
                <p style={{fontSize: '0.9em', color: '#888', marginTop: '10px'}}>
                    Want to refine? Edit the text above and click Generate.
                </p>
            </div>
        </div>
      )}
    </div>
  )
}

export default App
