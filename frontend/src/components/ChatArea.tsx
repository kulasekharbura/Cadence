import { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getConversation } from '../api/conversations';
import { sendMessage } from '../api/chat';
import type { SourceChunk } from '../api/chat';
import { Send, Loader2, Bot, User } from 'lucide-react';
import clsx from 'clsx';
import { SourcePanel } from './SourcePanel';
import { preprocessMessageContent } from '../utils/chatUtils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatAreaProps {
  conversationId: string | null;
}

export default function ChatArea({ conversationId }: ChatAreaProps) {
  const queryClient = useQueryClient();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [optimisticMessage, setOptimisticMessage] = useState<string | null>(null);
  const [isError, setIsError] = useState<boolean>(false);
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);

  const { data: conversation, isLoading } = useQuery({
    queryKey: ['conversation', conversationId],
    queryFn: () => getConversation(conversationId!),
    enabled: !!conversationId,
  });

  const chatMutation = useMutation({
    mutationFn: (msg: string) => sendMessage(conversationId!, msg),
    onSuccess: () => {
      setOptimisticMessage(null);
      setIsError(false);
      queryClient.invalidateQueries({ queryKey: ['conversation', conversationId] });
    },
    onError: () => {
      setIsError(true);
    }
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || !conversationId || chatMutation.isPending) return;

    const msg = inputValue.trim();
    setInputValue('');
    setOptimisticMessage(msg);
    setIsError(false);
    
    chatMutation.mutate(msg);
  };

  const handleRetry = () => {
    if (optimisticMessage) {
      setIsError(false);
      chatMutation.mutate(optimisticMessage);
    }
  };

  const handleCancelOptimistic = () => {
    setOptimisticMessage(null);
    setIsError(false);
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages, optimisticMessage, chatMutation.isPending]);

  if (!conversationId) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <Bot className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <p>Select or create a conversation to start chatting.</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  // Component to render custom source chips within ReactMarkdown
  const renderMarkdownComponents = (sources: SourceChunk[] = []) => ({
    a: ({ node, href, children, ...props }: any) => {
      if (href?.startsWith('#source-')) {
        const num = parseInt(href.replace('#source-', ''), 10);
        const source = sources.find(s => s.source_number === num);
        
        if (source) {
          return (
            <button
              key={`source-chip-${num}`}
              onClick={(e) => {
                e.preventDefault();
                setSelectedChunkId(source.chunk_id);
              }}
              title="Click to view source"
              data-testid={`source-chip-${num}`}
              className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 bg-blue-100 hover:bg-blue-200 text-blue-800 text-xs font-medium rounded transition-colors cursor-pointer"
            >
              {source.filename} &middot; p.{source.page_number}
            </button>
          );
        }
      }
      return <a href={href} {...props} className="text-blue-600 hover:underline">{children}</a>;
    }
  });

  return (
    <div className="flex flex-col h-full bg-white relative overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b bg-white flex items-center shadow-sm z-10">
        <h3 className="font-medium text-gray-800">
          {conversation?.title || 'Chat'}
        </h3>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {conversation?.messages.map((msg) => (
          <div key={msg.id} className={clsx("flex gap-4 max-w-3xl", msg.role === 'user' ? "ml-auto flex-row-reverse" : "")}>
            <div className={clsx(
              "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
              msg.role === 'user' ? "bg-blue-600" : "bg-teal-600"
            )}>
              {msg.role === 'user' ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-white" />}
            </div>
            
            <div className={clsx(
              "px-4 py-3 rounded-2xl whitespace-pre-wrap text-sm leading-relaxed",
              msg.role === 'user' ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800"
            )}>
              {msg.role === 'user' ? (
                msg.content
              ) : (
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={renderMarkdownComponents(msg.sources || [])}
                  >
                    {preprocessMessageContent(msg.content, msg.sources || [])}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Optimistic User Message */}
        {optimisticMessage && (
          <div className="flex gap-4 max-w-3xl ml-auto flex-row-reverse">
            <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-blue-600 opacity-70">
              <User className="w-5 h-5 text-white" />
            </div>
            
            <div className="flex flex-col items-end">
              <div className="px-4 py-3 rounded-2xl whitespace-pre-wrap text-sm bg-blue-600 text-white opacity-70">
                {optimisticMessage}
              </div>
              
              {isError && (
                <div className="flex items-center gap-2 mt-2 text-xs text-red-500">
                  <span>Failed to send.</span>
                  <button onClick={handleRetry} className="underline hover:text-red-700">Retry</button>
                  <button onClick={handleCancelOptimistic} className="underline hover:text-red-700">Cancel</button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Loading Assistant Message */}
        {chatMutation.isPending && !isError && (
          <div className="flex gap-4 max-w-3xl">
            <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-teal-600">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="px-4 py-3 rounded-2xl bg-gray-100 text-gray-800 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-teal-600" />
              <span className="text-sm text-gray-500">Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t bg-gray-50">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto relative flex items-center">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question about the documents..."
            disabled={chatMutation.isPending || isError}
            className="w-full pl-4 pr-12 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white shadow-sm disabled:bg-gray-100"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || chatMutation.isPending || isError}
            className="absolute right-2 p-2 rounded-lg text-blue-600 hover:bg-blue-50 disabled:opacity-50 disabled:hover:bg-transparent transition-colors"
          >
            {chatMutation.isPending && !isError ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </form>
      </div>
      
      <SourcePanel chunkId={selectedChunkId} onClose={() => setSelectedChunkId(null)} />
    </div>
  );
}
