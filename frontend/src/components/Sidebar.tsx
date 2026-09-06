import { useState, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getDocuments, uploadDocument, deleteDocument } from '../api/documents';
import type { Document } from '../api/documents';
import { listConversations, createConversation, deleteConversation } from '../api/conversations';
import { Upload, Check, Plus, MessageSquare, AlertCircle, Loader2, Trash2 } from 'lucide-react';
import clsx from 'clsx';
import { formatDistanceToNow } from 'date-fns';

interface SidebarProps {
  activeConversationId: string | null;
  onSelectConversation: (id: string | null) => void;
}

export default function Sidebar({ activeConversationId, onSelectConversation }: SidebarProps) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [deletingConvId, setDeletingConvId] = useState<string | null>(null);

  // Queries
  const { data: documents = [], isLoading: docsLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: getDocuments,
    // Poll every 3 seconds if any document is processing
    refetchInterval: (query) => {
      const data = query.state.data as Document[] | undefined;
      const isProcessing = data?.some(d => d.processing_status === 'UPLOADED' || d.processing_status === 'PROCESSING');
      return isProcessing ? 3000 : false;
    }
  });

  const { data: conversations = [], isLoading: convsLoading } = useQuery({
    queryKey: ['conversations'],
    queryFn: listConversations,
  });

  // Mutations
  const uploadMutation = useMutation({
    mutationFn: uploadDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setUploadError(null);
    },
    onError: (err: any) => {
      setUploadError(err.response?.data?.detail || 'Upload failed');
    }
  });

  const createConvMutation = useMutation({
    mutationFn: createConversation,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      onSelectConversation(data.id);
      setSelectedDocs(new Set()); // clear selection
    }
  });

  const deleteDocMutation = useMutation({
    mutationFn: deleteDocument,
    onMutate: (id) => setDeletingDocId(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setDeletingDocId(null);
      if (selectedDocs.has(id)) {
        const newSet = new Set(selectedDocs);
        newSet.delete(id);
        setSelectedDocs(newSet);
      }
    },
    onError: () => {
      setDeletingDocId(null);
    }
  });

  const deleteConvMutation = useMutation({
    mutationFn: deleteConversation,
    onMutate: (id) => setDeletingConvId(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      setDeletingConvId(null);
      if (activeConversationId === id) {
        onSelectConversation(null);
      }
    },
    onError: () => {
      setDeletingConvId(null);
    }
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setUploadError('Only PDF files are supported.');
        return;
      }
      uploadMutation.mutate(file);
    }
    // reset input
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const toggleDocSelection = (id: string) => {
    const newSet = new Set(selectedDocs);
    if (newSet.has(id)) newSet.delete(id);
    else newSet.add(id);
    setSelectedDocs(newSet);
  };

  const handleCreateConv = () => {
    if (selectedDocs.size > 0) {
      createConvMutation.mutate(Array.from(selectedDocs));
    }
  };

  const handleDeleteDoc = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (window.confirm("Delete this document?")) {
      deleteDocMutation.mutate(id);
    }
  };

  const handleDeleteConv = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (window.confirm("Delete this conversation?")) {
      deleteConvMutation.mutate(id);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Upload Section */}
      <div className="p-4 border-b">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Documents</h2>
        
        <input 
          type="file" 
          accept=".pdf" 
          ref={fileInputRef} 
          className="hidden" 
          onChange={handleFileChange}
        />
        
        <button 
          onClick={() => fileInputRef.current?.click()}
          disabled={uploadMutation.isPending}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors disabled:opacity-50"
        >
          {uploadMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
          Upload PDF
        </button>
        {uploadError && <p className="text-red-500 text-xs mt-2">{uploadError}</p>}
      </div>

      {/* Documents List */}
      <div className="flex-1 overflow-y-auto p-4 border-b">
        {docsLoading ? (
          <div className="flex justify-center p-4"><Loader2 className="w-5 h-5 animate-spin text-gray-400" /></div>
        ) : documents.length === 0 ? (
          <p className="text-sm text-gray-500 italic">No documents uploaded.</p>
        ) : (
          <div className="space-y-2">
            {documents.map(doc => {
              const isReady = doc.processing_status === 'READY';
              const isFailed = doc.processing_status === 'FAILED';
              const isProcessing = !isReady && !isFailed;
              const isSelected = selectedDocs.has(doc.id);

              return (
                <div 
                  key={doc.id}
                  onClick={() => isReady && toggleDocSelection(doc.id)}
                  className={clsx(
                    "flex items-center gap-3 p-2 rounded border cursor-pointer transition-colors text-sm group",
                    isReady ? (isSelected ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:bg-gray-50") : "border-gray-200 bg-gray-50 cursor-not-allowed opacity-75"
                  )}
                >
                  <div className="flex-shrink-0">
                    {isSelected ? (
                      <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center">
                        <Check className="w-3 h-3 text-white" />
                      </div>
                    ) : (
                      <div className="w-5 h-5 rounded-full border border-gray-300" />
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="truncate font-medium text-gray-800">{doc.filename}</p>
                    <div className="flex items-center gap-1 mt-0.5">
                      {isProcessing && <Loader2 className="w-3 h-3 animate-spin text-blue-500" />}
                      {isFailed && <AlertCircle className="w-3 h-3 text-red-500" />}
                      <span className={clsx(
                        "text-xs font-medium",
                        isProcessing && "text-blue-500",
                        isReady && "text-green-600",
                        isFailed && "text-red-500"
                      )}>
                        {doc.processing_status}
                      </span>
                    </div>
                  </div>
                  <button 
                    onClick={(e) => handleDeleteDoc(e, doc.id)}
                    disabled={deletingDocId === doc.id}
                    title="Delete document"
                    aria-label="Delete document"
                    className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-50 flex-shrink-0"
                  >
                    {deletingDocId === doc.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Conversations Section */}
      <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider">Conversations</h2>
          <button 
            onClick={handleCreateConv}
            disabled={selectedDocs.size === 0 || createConvMutation.isPending}
            className="p-1 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded disabled:opacity-50"
            title="Start new conversation with selected documents"
          >
            {createConvMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
          </button>
        </div>

        {convsLoading ? (
          <div className="flex justify-center p-4"><Loader2 className="w-5 h-5 animate-spin text-gray-400" /></div>
        ) : conversations.length === 0 ? (
          <p className="text-sm text-gray-500 italic">No conversations yet.</p>
        ) : (
          <div className="space-y-1">
            {conversations.map(conv => {
              const isActive = conv.id === activeConversationId;
              return (
                <button
                  key={conv.id}
                  onClick={() => onSelectConversation(conv.id)}
                  className={clsx(
                    "w-full flex items-center justify-between p-3 rounded-md text-left transition-colors text-sm group",
                    isActive ? "bg-white shadow-sm border border-gray-200" : "hover:bg-gray-200 border border-transparent"
                  )}
                >
                  <div className="flex flex-col min-w-0 flex-1">
                    <div className="flex items-center gap-2 w-full">
                      <MessageSquare className="w-4 h-4 text-gray-400 flex-shrink-0" />
                      <span className="font-medium text-gray-800 truncate">{conv.title || 'Untitled'}</span>
                    </div>
                    <span className="text-xs text-gray-500 mt-1 pl-6">
                      {formatDistanceToNow(new Date(conv.created_at), { addSuffix: true })}
                    </span>
                  </div>
                  <div 
                    onClick={(e) => handleDeleteConv(e, conv.id)}
                    role="button"
                    tabIndex={0}
                    title="Delete conversation"
                    aria-label="Delete conversation"
                    className={clsx(
                      "p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-opacity flex-shrink-0 ml-2",
                      (isActive || deletingConvId === conv.id) ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                    )}
                  >
                    {deletingConvId === conv.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
