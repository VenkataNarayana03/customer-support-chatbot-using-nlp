import React, { useState } from 'react';
import './ChatInput.css';
import { Send, Trash2 } from 'lucide-react';

const ChatInput = ({ onSendMessage, onClearChat }) => {
  const [message, setMessage] = useState('');

  const handleSend = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSend(e);
    }
  };

  return (
    <div className="chat-input-container">
      <form className="chat-input-form" onSubmit={handleSend}>

        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          className="chat-textarea"
          rows={1}
        />
        <button type="submit" className="send-btn" disabled={!message.trim()}>
          <Send size={18} />
        </button>
        <button type="button" className="clear-btn" onClick={onClearChat} title="Clear Chat">
          <Trash2 size={18} />
        </button>
      </form>
    </div>
  );
};

export default ChatInput;
