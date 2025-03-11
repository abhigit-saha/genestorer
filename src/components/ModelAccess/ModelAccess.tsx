import React, { useState } from 'react';
import { Brain, Upload, TrendingUp } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export const ModelAccess = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [epoch, setEpoch] = useState(0);

  const trainingData = {
    labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'],
    datasets: [
      {
        label: 'Training Accuracy',
        data: [0.65, 0.75, 0.82, 0.87, 0.89],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
      },
      {
        label: 'Validation Accuracy',
        data: [0.63, 0.72, 0.78, 0.82, 0.84],
        borderColor: 'rgb(147, 51, 234)',
        backgroundColor: 'rgba(147, 51, 234, 0.5)',
      },
    ],
  };

  return (
    <div className="space-y-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Model Access & Training</h1>
        <p className="mt-2 text-gray-600">
          Access AI models, contribute to training, and improve model accuracy
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center mb-6">
            <Brain className="h-6 w-6 text-blue-600 mr-2" />
            <h2 className="text-xl font-semibold">Model Selection</h2>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Choose a model...</option>
                <option value="pneumonia">Pneumonia Detection</option>
                <option value="diabetes">Diabetes Prediction</option>
                <option value="heart">Heart Disease Analysis</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Training Parameters
              </label>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-500">Learning Rate</label>
                  <input
                    type="number"
                    step="0.001"
                    defaultValue="0.001"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500">Batch Size</label>
                  <input
                    type="number"
                    defaultValue="32"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  />
                </div>
              </div>
            </div>

            <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 flex items-center justify-center">
              <Upload className="h-5 w-5 mr-2" />
              Start Training
            </button>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center mb-6">
            <TrendingUp className="h-6 w-6 text-purple-600 mr-2" />
            <h2 className="text-xl font-semibold">Training Progress</h2>
          </div>
          
          <div className="h-64">
            <Line
              data={trainingData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'bottom',
                  },
                },
              }}
            />
          </div>

          <div className="mt-4 p-4 bg-gray-50 rounded-md">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Current Epoch</p>
                <p className="font-semibold">{epoch}/100</p>
              </div>
              <div>
                <p className="text-gray-500">Training Accuracy</p>
                <p className="font-semibold">89.2%</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};