import React, { useState } from 'react';
import { FileUploadPanel } from './FileUploadPanel';
import { StoragePanel } from './StoragePanel';
import { UploadProgress } from './UploadProgress';

export const InstitutionalStorage = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<'uploading' | 'encrypted' | 'failed'>('uploading');

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setUploadProgress(0);
    setUploadStatus('uploading');
    
    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setUploadStatus('encrypted');
          return 100;
        }
        return prev + 10;
      });
    }, 500);
  };

  return (
    <div className="space-y-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Institutional Data Storage</h1>
        <p className="mt-2 text-gray-600">
          Securely upload and manage your medical data files with enterprise-grade encryption
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <FileUploadPanel onFileSelect={handleFileSelect} />
        {selectedFile && (
          <UploadProgress
            progress={uploadProgress}
            status={uploadStatus}
            fileDetails={{
              name: selectedFile.name,
              size: selectedFile.size,
              format: selectedFile.name.split('.').pop()?.toUpperCase() || 'Unknown',
              uploadDate: new Date(),
            }}
          />
        )}
      </div>

      <StoragePanel />
    </div>
  );
};