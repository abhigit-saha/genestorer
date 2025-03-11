import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X } from 'lucide-react';

const ALLOWED_FILE_TYPES = ['.vcf', '.pdf', '.csv'];

interface FileUploadPanelProps {
  onFileSelect: (file: File) => void;
}

export const FileUploadPanel: React.FC<FileUploadPanelProps> = ({ onFileSelect }) => {
  const [selectedFileType, setSelectedFileType] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/pdf': ['.pdf'],
      'text/vcard': ['.vcf'],
    },
    maxFiles: 1,
  });

  const removeFile = () => {
    setSelectedFile(null);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select File Type
        </label>
        <select
          value={selectedFileType}
          onChange={(e) => setSelectedFileType(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select type...</option>
          {ALLOWED_FILE_TYPES.map((type) => (
            <option key={type} value={type}>
              {type.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-600">
          Drag & drop your file here, or{' '}
          <span className="text-blue-500 font-medium">browse</span>
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Supported formats: VCF, PDF, CSV
        </p>
      </div>

      {selectedFile && (
        <div className="mt-4 p-4 bg-gray-50 rounded-md">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <File className="h-5 w-5 text-gray-500 mr-2" />
              <span className="text-sm font-medium text-gray-700">
                {selectedFile.name}
              </span>
            </div>
            <button
              onClick={removeFile}
              className="text-gray-400 hover:text-gray-500"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};