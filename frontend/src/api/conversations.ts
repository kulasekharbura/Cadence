import { apiClient } from './apiClient';
import type { SourceChunk } from './chat';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  sources?: SourceChunk[];
}

export interface Conversation {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  document_ids: string[];
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export const listConversations = async (): Promise<Conversation[]> => {
  const { data } = await apiClient.get<Conversation[]>('/conversations');
  return data;
};

export const getConversation = async (id: string): Promise<ConversationDetail> => {
  const { data } = await apiClient.get<ConversationDetail>(`/conversations/${id}`);
  return data;
};

export const createConversation = async (documentIds: string[]): Promise<Conversation> => {
  const { data } = await apiClient.post<Conversation>('/conversations', { document_ids: documentIds });
  return data;
};

export const deleteConversation = async (id: string): Promise<void> => {
  await apiClient.delete(`/conversations/${id}`);
};
