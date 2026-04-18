import React, { useState, useMemo } from 'react';
import ChatLayout from './components/ChatLayout';
import MessageBubble from './components/MessageBubble';
import ChatInput from './components/ChatInput';
import { sendChatMessage } from './api/chatApi';

function App() {
  const sessionId = useMemo(() => `session-${Date.now()}`, []);
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! Im your AI support assistant. How can I help you today?",
      isBot: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  const handleSendMessage = async (text) => {
    const userMsg = {
      id: Date.now(),
      text,
      isBot: false,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setMessages(prev => [...prev, userMsg]);
    setIsTyping(true);

    try {
      const data = await sendChatMessage(text, sessionId);
      const botMsg = {
        id: Date.now() + 1,
        text: data.response,
        isBot: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, botMsg]);
    } catch {
      const errorMsg = {
        id: Date.now() + 1,
        text: "I couldn't reach the support server. Please make sure the backend is running.",
        isBot: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const toggleFullScreen = () => {
    setIsFullScreen(!isFullScreen);
  };

  const toggleMinimize = () => {
    if (isFullScreen) {
      setIsFullScreen(false);
    }
    setIsMinimized(!isMinimized);
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: Date.now(),
        text: "Hello! Im your AI support assistant. How can I help you today?",
        isBot: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
  };

  return (
    <ChatLayout 
      isFullScreen={isFullScreen} 
      toggleFullScreen={toggleFullScreen}
      isMinimized={isMinimized}
      toggleMinimize={toggleMinimize}
      chatInput={<ChatInput onSendMessage={handleSendMessage} onClearChat={handleClearChat} />}
    >
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          message={msg.text}
          isBot={msg.isBot}
          timestamp={msg.timestamp}
        />
      ))}
      {isTyping && (
        <div style={{ display: 'flex', gap: '8px', padding: '16px 0', color: 'var(--text-secondary)'}}>
          <BotTypingIndicator />
        </div>
      )}
    </ChatLayout>
  );
}

const BotTypingIndicator = () => (
  <div className="message-wrapper bot">
    <div className="message-avatar bot" style={{ width: '28px', height: '28px' }}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-bot"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
    </div>
    <div className="message-bubble bot-bubble" style={{ display: 'flex', gap: '4px', alignItems: 'center', padding: '12px 16px' }}>
      <span className="dot" style={dotStyle(0)}></span>
      <span className="dot" style={dotStyle(0.2)}></span>
      <span className="dot" style={dotStyle(0.4)}></span>
    </div>
    <style>{`
      @keyframes typing {
        0%, 100% { transform: translateY(0); opacity: 0.4; }
        50% { transform: translateY(-4px); opacity: 1; }
      }
    `}</style>
  </div>
);

const dotStyle = (delay) => ({
  width: '6px',
  height: '6px',
  backgroundColor: 'var(--text-primary)',
  borderRadius: '50%',
  animation: 'typing 1.4s infinite ease-in-out',
  animationDelay: delay + 's'
});

export default App;
