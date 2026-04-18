import React, { useEffect, useRef, useState } from 'react';
import './ChatLayout.css';
import { Bot, Maximize2, Minimize2, MoreVertical, Minus } from 'lucide-react';

const ChatLayout = ({ children, isFullScreen, toggleFullScreen, isMinimized, toggleMinimize, chatInput }) => {
  const messagesEndRef = useRef(null);
  
  // Dragging state
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef({ x: 0, y: 0 });
  
  // Resizing state
  const [width, setWidth] = useState(460);
  const [resizingDir, setResizingDir] = useState(null);
  const resizeStart = useRef({ x: 0, width: 0, posX: 0 });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [children]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isDragging) {
        setPosition({
          x: e.clientX - dragStartPos.current.x,
          y: e.clientY - dragStartPos.current.y
        });
      } else if (resizingDir) {
        const deltaX = e.clientX - resizeStart.current.x;
        if (resizingDir === 'right') {
          setWidth(Math.max(320, resizeStart.current.width + deltaX));
        } else if (resizingDir === 'left') {
          const newWidth = Math.max(320, resizeStart.current.width - deltaX);
          const diff = newWidth - resizeStart.current.width;
          setWidth(newWidth);
          setPosition(prev => ({ ...prev, x: resizeStart.current.posX - diff }));
        }
      }
    };

    const handleMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
        document.body.style.userSelect = '';
      }
      if (resizingDir) {
        setResizingDir(null);
        document.body.style.userSelect = '';
      }
    };

    if (isDragging || resizingDir) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, resizingDir]);

  const handleMouseDown = (e) => {
    if (isFullScreen) return;
    
    // Prevent drag if clicking on interactive elements
    const isInteractive = e.target.closest('button, input, textarea, .chat-messages');
    if (isInteractive) return;

    setIsDragging(true);
    document.body.style.userSelect = 'none'; // Prevent text selection while dragging
    dragStartPos.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    };
  };

  const handleResizeDown = (e, dir) => {
    e.stopPropagation();
    if (isFullScreen) return;
    setResizingDir(dir);
    document.body.style.userSelect = 'none';
    resizeStart.current = {
      x: e.clientX,
      width: width,
      posX: position.x
    };
  };

  return (
    <div 
      className={`chat-container glass-panel ${isFullScreen ? 'full-screen' : ''} ${isMinimized ? 'minimized' : ''}`}
      onMouseDown={handleMouseDown}
      style={{ 
        width: isFullScreen ? '100vw' : `${width}px`,
        transform: isFullScreen ? 'none' : `translate(${position.x}px, ${position.y}px)`,
        cursor: isFullScreen ? 'default' : (isDragging ? 'grabbing' : 'grab'),
        transition: (isDragging || resizingDir) ? 'none' : 'all 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
        position: 'relative'
      }}
    >
      {!isFullScreen && (
        <>
          <div className="resize-handle left" onMouseDown={(e) => handleResizeDown(e, 'left')} />
          <div className="resize-handle right" onMouseDown={(e) => handleResizeDown(e, 'right')} />
        </>
      )}
      <div className="chat-header">
        <div className="chat-header-info">
          <div className="bot-avatar glass-panel">
            <Bot size={20} color="var(--text-primary)" />
          </div>
          <div>
            <h2 className="bot-name">Support Assistant</h2>
            <div className="status-indicator">
              <span className="status-dot"></span>
              <span className="status-text">Online</span>
            </div>
          </div>
        </div>
        <div className="chat-header-actions">
          <button 
            className="icon-btn" 
            onClick={toggleFullScreen}
            title={isFullScreen ? "Minimize" : "Maximize"}
          >
            {isFullScreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
          <button 
            className="icon-btn" 
            onClick={toggleMinimize}
            title="Minimize"
          >
            <Minus size={18} />
          </button>
          <button className="icon-btn" title="More Options">
            <MoreVertical size={18} />
          </button>
        </div>
      </div>
      
      <div className="chat-messages">
        {children}
        <div ref={messagesEndRef} />
      </div>
      {chatInput}
    </div>
  );
};

export default ChatLayout;
