import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navigation } from './components/Navigation';
import { WelcomePage } from './components/Welcome/WelcomePage';
import { InstitutionalStorage } from './components/InstitutionalStorage/InstitutionalStorage';
import { ModelAccess } from './components/ModelAccess/ModelAccess';
import { DiseaseDetection } from './components/DiseaseDetection/DiseaseDetection';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<WelcomePage />} />
            <Route path="/storage" element={<InstitutionalStorage />} />
            <Route path="/model-access" element={<ModelAccess />} />
            <Route path="/disease-detection" element={<DiseaseDetection />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;