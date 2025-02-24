import logging
import sys
import streamlit as st
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to stdout (terminal)
    ]
)

logger = logging.getLogger('trash_detection')

# In-app debug logs
debug_logs = []

def log_debug(message):
    """Log debug message to terminal and store for UI display"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    formatted_msg = f"{timestamp} [DEBUG] {message}"
    logger.debug(message)
    debug_logs.append(formatted_msg)
    return message

def log_info(message):
    """Log info message to terminal and store for UI display"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    formatted_msg = f"{timestamp} [INFO] {message}"
    logger.info(message)
    debug_logs.append(formatted_msg)
    return message

def log_error(message):
    """Log error message to terminal and store for UI display"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    formatted_msg = f"{timestamp} [ERROR] {message}"
    logger.error(message)
    debug_logs.append(formatted_msg)
    return message

def show_debug_panel():
    """Display debug panel in the Streamlit UI"""
    with st.expander("Debug Logs", expanded=False):
        show_terminal = st.checkbox("Show Live Debug Logs", value=False)
        
        if show_terminal:
            # Create a container for logs
            log_container = st.container()
            
            # Display logs in reverse order (newest first)
            with log_container:
                for log in reversed(debug_logs[-50:]):  # Show last 50 logs
                    st.text(log)
                    
            # Add clear button
            if st.button("Clear Logs"):
                debug_logs.clear() 