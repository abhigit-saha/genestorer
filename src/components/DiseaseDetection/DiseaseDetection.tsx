import React, { useState, useEffect } from 'react';
import { Activity, AlertCircle, CheckCircle } from 'lucide-react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export const DiseaseDetection = () => {
  const [formData, setFormData] = useState({
    age: '45',
    sex: '1',
    chest_pain_type: '3',
    resting_blood_pressure: '130',
    cholesterol: '250',
    fasting_blood_sugar: '0',
    resting_ecg_results: '1',
    max_heart_rate: '175',
    exercise_induced_angina: '0',
    st_depression: '1.2',
    slope_st_segment: '2',
    major_vessels: '0',
    thalassemia: '3',
    region: '2',
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);

  useEffect(() => {
    async function fetchGraph() {
      try {
        const response = await fetch("http://127.0.0.1:8000/regional-insights");
        if (response.ok) {
          setImageSrc("http://127.0.0.1:8000/regional-insights");
        } else {
          console.error("Graph not found");
        }
      } catch (error) {
        console.error("Error fetching graph:", error);
      }
    }
    fetchGraph();
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const features = Object.values(formData).map(Number);
    
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold text-gray-900">Disease Detection</h1>
      <p className="text-gray-600">AI-powered disease prediction & regional health insights</p>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-xl shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Patient Information</h2>
          <form className="space-y-4" onSubmit={handleSubmit}>
            {Object.keys(formData).map((key, index) => (
              <div key={index}>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </label>
                <input
                  type="number"
                  name={key}
                  value={formData[key]}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
            ))}
            <button
              type="submit"
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
            >
              {loading ? 'Analyzing...' : 'Analyze Symptoms'}
            </button>
          </form>
        </div>

        <div className="space-y-8">
          {prediction && (
            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
              <div className={prediction.prediction ? "bg-red-50 p-4 rounded-lg" : "bg-green-50 p-4 rounded-lg"}>
                <div className="flex items-center mb-2">
                  {prediction.prediction ? (
                    <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
                  ) : (
                    <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                  )}
                  <h3 className={`font-semibold ${prediction.prediction ? 'text-red-900' : 'text-green-900'}`}>
                    {prediction.prediction ? 'High Risk' : 'Low Risk'}
                  </h3>
                </div>
                <p className={prediction.prediction ? "text-red-700" : "text-green-700"}>
                  Probability: {(prediction.probability * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          )}

          <div className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Regional Insights</h2>
            <div className="h-64 flex items-center justify-center">
              {imageSrc ? (
                <img src={imageSrc} alt="Probability of Disease vs Region" className="w-full h-full object-contain" />
              ) : (
                <p className="text-gray-600">Graph not available. Train the model first.</p>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};
