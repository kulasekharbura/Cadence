import { apiClient } from './apiClient';

export interface Document {
  id: string;
  filename: string;
  file_path: string;
  file_size: number;
  page_count: number | null;
  processing_status: 'UPLOADED' | 'PROCESSING' | 'READY' | 'FAILED';
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export const getDocuments = async (): Promise<Document[]> => {
  const { data } = await apiClient.get<Document[]>('/documents');
  return data;
};

export const uploadDocument = async (file: File): Promise<Document> => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post<Document>('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return data;
};

export const deleteDocument = async (id: string): Promise<void> => {
  await apiClient.delete(`/documents/${id}`);
};
