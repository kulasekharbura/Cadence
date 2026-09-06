import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';

const queryClient = new QueryClient();

function App() {
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex h-screen bg-gray-50 font-sans">
        {/* Left Sidebar */}
        <div className="w-80 border-r bg-white flex flex-col h-full flex-shrink-0">
          <Sidebar
            activeConversationId={activeConversationId}
            onSelectConversation={setActiveConversationId}
          />
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full overflow-hidden relative">
          <ChatArea conversationId={activeConversationId} />
        </div>
      </div>
    </QueryClientProvider>
  );
}

export default App;
