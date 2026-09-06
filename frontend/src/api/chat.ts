import { apiClient } from './apiClient';

export interface SourceChunk {
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_index: number;
  source_number: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceChunk[];
}

export const sendMessage = async (conversationId: string, message: string): Promise<ChatResponse> => {
  const { data } = await apiClient.post<ChatResponse>('/chat', {
    conversation_id: conversationId,
    message,
  });
  return data;
};
