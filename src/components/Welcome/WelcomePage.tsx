import React from 'react';
import { Link } from 'react-router-dom';
import { Database, Brain, Stethoscope, ArrowRight } from 'lucide-react';

export const WelcomePage = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Secure Health Data System
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          A comprehensive platform for secure medical data management, AI model training,
          and disease prediction
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <div className="flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-6">
            <Database className="h-8 w-8 text-blue-600" />
          </div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">
            Institutional Data Storage
          </h2>
          <p className="text-gray-600 mb-6">
            Securely upload, encrypt, and manage medical data with our advanced storage system.
          </p>
          <Link
            to="/storage"
            className="inline-flex items-center text-blue-600 hover:text-blue-700"
          >
            Access Storage
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <div className="flex items-center justify-center w-16 h-16 bg-purple-100 rounded-full mb-6">
            <Brain className="h-8 w-8 text-purple-600" />
          </div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">
            Model Access & Training
          </h2>
          <p className="text-gray-600 mb-6">
            Access AI models, contribute to training, and submit weight updates for improved accuracy.
          </p>
          <Link
            to="/model-access"
            className="inline-flex items-center text-purple-600 hover:text-purple-700"
          >
            Access Models
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <div className="flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-6">
            <Stethoscope className="h-8 w-8 text-green-600" />
          </div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">
            Disease Detection
          </h2>
          <p className="text-gray-600 mb-6">
            Utilize AI-powered disease prediction and gain valuable regional health insights.
          </p>
          <Link
            to="/disease-detection"
            className="inline-flex items-center text-green-600 hover:text-green-700"
          >
            Start Detection
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>
      </div>

      <div className="mt-16 bg-blue-50 rounded-2xl p-8">
        <h3 className="text-2xl font-semibold text-gray-900 mb-4 text-center">
          Why Choose SHDS?
        </h3>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <h4 className="font-semibold text-gray-900 mb-2">Secure & Compliant</h4>
            <p className="text-gray-600">
              Enterprise-grade encryption and HIPAA-compliant data management
            </p>
          </div>
          <div className="text-center">
            <h4 className="font-semibold text-gray-900 mb-2">AI-Powered Insights</h4>
            <p className="text-gray-600">
              Advanced machine learning models for accurate disease prediction
            </p>
          </div>
          <div className="text-center">
            <h4 className="font-semibold text-gray-900 mb-2">Collaborative Platform</h4>
            <p className="text-gray-600">
              Connect with institutions and contribute to model improvement
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};