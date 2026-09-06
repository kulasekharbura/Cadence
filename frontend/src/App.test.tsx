import '@testing-library/jest-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from './App';
import * as docsApi from './api/documents';
import * as convsApi from './api/conversations';
import * as chatApi from './api/chat';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock API modules
vi.mock('./api/documents');
vi.mock('./api/conversations');
vi.mock('./api/chat');

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

describe('App Integration Tests', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  const renderApp = () => {
    const queryClient = createTestQueryClient();
    return render(
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    );
  };

  it('renders documents and conversation list', async () => {
    vi.mocked(docsApi.getDocuments).mockResolvedValue([
      { id: '1', filename: 'test.pdf', processing_status: 'READY' } as any
    ]);
    vi.mocked(convsApi.listConversations).mockResolvedValue([
      { id: 'c1', title: 'Test Conv', created_at: new Date().toISOString() } as any
    ]);

    renderApp();
    
    // Wait for async fetch
    expect(await screen.findByText('test.pdf')).toBeInTheDocument();
    expect(await screen.findByText('Test Conv')).toBeInTheDocument();
  });

  it('handles document selection and conversation creation', async () => {
    vi.mocked(docsApi.getDocuments).mockResolvedValue([
      { id: '1', filename: 'doc_ready.pdf', processing_status: 'READY' } as any
    ]);
    vi.mocked(convsApi.listConversations).mockResolvedValue([]);
    vi.mocked(convsApi.createConversation).mockResolvedValue({ id: 'cnew', title: 'New', created_at: new Date().toISOString() } as any);
    vi.mocked(convsApi.getConversation).mockResolvedValue({ id: 'cnew', messages: [], created_at: new Date().toISOString() } as any);

    renderApp();
    
    // Select document
    const docItem = await screen.findByText('doc_ready.pdf');
    fireEvent.click(docItem);
    
    // Click create
    const createBtn = screen.getByTitle('Start new conversation with selected documents');
    expect(createBtn).not.toBeDisabled();
    fireEvent.click(createBtn);
    
    await waitFor(() => {
      expect(convsApi.createConversation).toHaveBeenCalledWith(['1'], expect.anything());
    });
  });

  it('shows unselectable processing status', async () => {
    vi.mocked(docsApi.getDocuments).mockResolvedValue([
      { id: '1', filename: 'doc_proc.pdf', processing_status: 'PROCESSING' } as any
    ]);
    vi.mocked(convsApi.listConversations).mockResolvedValue([]);

    renderApp();
    
    const docItem = await screen.findByText('doc_proc.pdf');
    fireEvent.click(docItem);
    
    // Create button should still be disabled because we couldn't select it
    const createBtn = screen.getByTitle('Start new conversation with selected documents');
    expect(createBtn).toBeDisabled();
  });

  it('sends chat message and renders optimistic UI then final response with sources', async () => {
    vi.mocked(docsApi.getDocuments).mockResolvedValue([]);
    vi.mocked(convsApi.listConversations).mockResolvedValue([{ id: 'c1', title: 'Chat', created_at: new Date().toISOString() } as any]);
    vi.mocked(convsApi.getConversation).mockResolvedValue({ id: 'c1', messages: [], created_at: new Date().toISOString() } as any);
    
    let resolveChat: any;
    const chatPromise = new Promise((res) => { resolveChat = res; });
    vi.mocked(chatApi.sendMessage).mockImplementation(() => chatPromise as any);

    // After chat is resolved, invalidate triggers getConversation again. Let's mock it to return the new message
    vi.mocked(convsApi.getConversation).mockImplementation(async () => {
      // If chat is resolved, return the message, else return empty
      if (vi.mocked(chatApi.sendMessage).mock.calls.length > 0) {
        return { 
          id: 'c1', 
          created_at: new Date().toISOString(),
          messages: [
            { id: 'm1', role: 'user', content: 'Hello', created_at: new Date().toISOString() },
            { 
              id: 'm2', 
              role: 'assistant', 
              content: 'This is the answer from [Source 1].', 
              created_at: new Date().toISOString(),
              sources: [{ source_number: 1, chunk_id: '123', document_id: '456', filename: 'doc.pdf', page_number: 5, chunk_index: 0 }]
            }
          ]
        } as any;
      }
      return { id: 'c1', messages: [], created_at: new Date().toISOString() } as any;
    });

    renderApp();
    
    // Select conversation
    fireEvent.click(await screen.findByText('Chat'));
    
    // Type and send message
    const input = await screen.findByPlaceholderText('Ask a question about the documents...');
    fireEvent.change(input, { target: { value: 'Hello' } });
    
    const sendBtn = input.nextElementSibling as HTMLButtonElement;
    fireEvent.click(sendBtn);
    
    // Optimistic UI should show 'Hello' and disable input
    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(input).toBeDisabled();
    
    // Resolve chat
    resolveChat({
      answer: 'This is the answer from [Source 1].',
      sources: [{ source_number: 1, chunk_id: '123', document_id: '456', filename: 'doc.pdf', page_number: 5, chunk_index: 0 }]
    });
    
    // The citation should render
    await waitFor(() => {
      expect(screen.getByText('doc.pdf · p.5')).toBeInTheDocument();
    });
    expect(screen.getByText(/This is the answer from/)).toBeInTheDocument();
  });

  it('handles chat failure and allows retry or cancel', async () => {
    vi.mocked(docsApi.getDocuments).mockResolvedValue([]);
    vi.mocked(convsApi.listConversations).mockResolvedValue([{ id: 'c1', title: 'Chat', created_at: new Date().toISOString() } as any]);
    vi.mocked(convsApi.getConversation).mockResolvedValue({ id: 'c1', messages: [], created_at: new Date().toISOString() } as any);
    
    vi.mocked(chatApi.sendMessage).mockRejectedValue(new Error('Failed'));

    renderApp();
    
    // Select conversation
    fireEvent.click(await screen.findByText('Chat'));
    
    // Send message
    const input = await screen.findByPlaceholderText('Ask a question about the documents...');
    fireEvent.change(input, { target: { value: 'FailMsg' } });
    fireEvent.click(input.nextElementSibling as HTMLButtonElement);
    
    // Should show error and retry
    expect(await screen.findByText('Failed to send.')).toBeInTheDocument();
    
    const retryBtn = screen.getByText('Retry');
    const cancelBtn = screen.getByText('Cancel');
    
    expect(retryBtn).toBeInTheDocument();
    
    // Click cancel
    fireEvent.click(cancelBtn);
    expect(screen.queryByText('FailMsg')).not.toBeInTheDocument();
  });

  it('handles document and conversation deletion', async () => {
    vi.mocked(docsApi.getDocuments).mockResolvedValue([
      { id: 'd1', filename: 'delete_me.pdf', processing_status: 'READY' } as any
    ]);
    vi.mocked(convsApi.listConversations).mockResolvedValue([
      { id: 'c1', title: 'Conv to Delete', created_at: new Date().toISOString() } as any
    ]);
    vi.mocked(docsApi.deleteDocument).mockResolvedValue();
    vi.mocked(convsApi.deleteConversation).mockResolvedValue();

    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

    renderApp();
    
    const deleteDocBtn = await screen.findByTitle('Delete document');
    const deleteConvBtn = await screen.findByTitle('Delete conversation');
    
    fireEvent.click(deleteDocBtn);
    expect(confirmSpy).toHaveBeenCalledWith('Delete this document?');
    await waitFor(() => {
      expect(docsApi.deleteDocument).toHaveBeenCalled();
    });
    
    fireEvent.click(deleteConvBtn);
    expect(confirmSpy).toHaveBeenCalledWith('Delete this conversation?');
    await waitFor(() => {
      expect(convsApi.deleteConversation).toHaveBeenCalled();
    });
    
    confirmSpy.mockRestore();
  });
});
