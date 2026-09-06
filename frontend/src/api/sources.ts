import { useQuery } from '@tanstack/react-query';
import { apiClient } from './apiClient';

export interface ChunkResponse {
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_index: number;
  content: string;
}

export const fetchSourceChunk = async (chunkId: string): Promise<ChunkResponse> => {
  const { data } = await apiClient.get<ChunkResponse>(`/sources/${chunkId}`);
  return data;
};

export const useSourceChunk = (chunkId: string | null) => {
  return useQuery({
    queryKey: ['source', chunkId],
    queryFn: () => chunkId ? fetchSourceChunk(chunkId) : Promise.reject('No chunkId'),
    enabled: !!chunkId,
  });
};
