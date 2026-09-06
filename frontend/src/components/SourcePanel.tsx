import React from 'react';
import { useSourceChunk } from '../api/sources';

interface SourcePanelProps {
  chunkId: string | null;
  onClose: () => void;
}

export const SourcePanel: React.FC<SourcePanelProps> = ({ chunkId, onClose }) => {
  const { data: chunk, isLoading, error } = useSourceChunk(chunkId);

  if (!chunkId) return null;

  return (
    <div className="w-80 border-l border-gray-200 bg-gray-50 flex flex-col h-full absolute right-0 top-0 bottom-0 shadow-lg z-10 transition-transform overflow-hidden">
      <div className="p-4 border-b border-gray-200 bg-white flex justify-between items-center sticky top-0">
        <h3 className="font-semibold text-gray-800">Source Details</h3>
        <button 
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 focus:outline-none p-1 rounded-md hover:bg-gray-100"
          aria-label="Close"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>

      <div className="flex-1 p-4 overflow-y-auto">
        {isLoading && (
          <div className="flex justify-center items-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        )}

        {error && (
          <div className="text-red-500 bg-red-50 p-3 rounded-md text-sm border border-red-200">
            Failed to load source chunk. It may have been deleted.
          </div>
        )}

        {chunk && !isLoading && !error && (
          <div className="flex flex-col space-y-4">
            <div className="bg-white p-3 rounded-md border border-gray-200 shadow-sm">
              <div className="text-xs text-gray-500 mb-1 font-semibold uppercase tracking-wider">Document</div>
              <div className="text-sm font-medium text-gray-800 break-words">{chunk.filename}</div>
            </div>
            
            <div className="bg-white p-3 rounded-md border border-gray-200 shadow-sm">
              <div className="text-xs text-gray-500 mb-1 font-semibold uppercase tracking-wider">Page</div>
              <div className="text-sm font-medium text-gray-800">{chunk.page_number}</div>
            </div>

            <div className="bg-white p-4 rounded-md border border-gray-200 shadow-sm flex-1">
              <div className="text-xs text-gray-500 mb-2 font-semibold uppercase tracking-wider">Content</div>
              <div className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                {chunk.content}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
