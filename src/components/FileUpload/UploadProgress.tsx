import React from 'react';
import { CheckCircle, AlertCircle, Clock } from 'lucide-react';

interface UploadProgressProps {
  progress: number;
  status: 'uploading' | 'encrypted' | 'failed';
  fileDetails: {
    name: string;
    size: number;
    format: string;
    uploadDate: Date;
  };
}

export const UploadProgress: React.FC<UploadProgressProps> = ({
  progress,
  status,
  fileDetails,
}) => {
  const statusConfig = {
    uploading: {
      color: 'bg-blue-500',
      icon: Clock,
      text: 'Uploading...',
    },
    encrypted: {
      color: 'bg-green-500',
      icon: CheckCircle,
      text: 'Encrypted',
    },
    failed: {
      color: 'bg-red-500',
      icon: AlertCircle,
      text: 'Failed',
    },
  };

  const { color, icon: StatusIcon, text } = statusConfig[status];

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center">
            <StatusIcon
              className={`h-5 w-5 ${
                status === 'uploading'
                  ? 'text-blue-500'
                  : status === 'encrypted'
                  ? 'text-green-500'
                  : 'text-red-500'
              } mr-2`}
            />
            <span className="font-medium">{text}</span>
          </div>
          <span className="text-sm text-gray-500">{progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`${color} h-2 rounded-full transition-all duration-300`}
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">File Name:</span>
          <span className="font-medium">{fileDetails.name}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Size:</span>
          <span className="font-medium">
            {(fileDetails.size / 1024 / 1024).toFixed(2)} MB
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Format:</span>
          <span className="font-medium">{fileDetails.format}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Upload Date:</span>
          <span className="font-medium">
            {fileDetails.uploadDate.toLocaleDateString()}
          </span>
        </div>
      </div>
    </div>
  );
};