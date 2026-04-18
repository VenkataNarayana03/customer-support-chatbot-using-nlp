import React from 'react';
import './MessageBubble.css';
import { Bot, User } from 'lucide-react';

const MessageBubble = ({ message, isBot, timestamp }) => {
  return (
    <div className={`message-wrapper ${isBot ? 'bot' : 'user'}`}>
      {isBot && (
        <div className="message-avatar bot">
          <Bot size={16} />
        </div>
      )}
      <div className="message-content-container">
        <div className={`message-bubble ${isBot ? 'bot-bubble' : 'user-bubble'}`}>
          <p>{message}</p>
        </div>
        <span className="message-timestamp">{timestamp}</span>
      </div>
      {!isBot && (
        <div className="message-avatar user">
          <User size={16} />
        </div>
      )}
    </div>
  );
};

export default MessageBubble;
