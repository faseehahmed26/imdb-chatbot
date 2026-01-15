"""
Telegram Bot for IMDB Conversational Agent
Integrates with the LangGraph agent workflow
"""

from config import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY
from agents import query_agent
import os
import sys
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction, ParseMode

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))


# =============================================================================
# COMMAND HANDLERS
# =============================================================================


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_message = """
**Welcome to IMDB Movie Assistant!**

I can help you explore the IMDB Top 1000 movies dataset. Ask me anything about movies!

**Example Questions:**
• When did The Matrix release?
• Top 5 movies of 2019
• Horror movies with high ratings
• Movies with police themes
• Steven Spielberg sci-fi films

**Commands:**
/start - Show this message
/help - Show help information
/examples - Show more example questions

Just type your question and I'll help you find the answer!
    """
    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
**IMDB Movie Assistant Help**

**What I can do:**
• Find movies by year, rating, genre, director
• Search for movies with specific themes or plots
• Answer questions about top movies and rankings
• Provide movie details and summaries

**Query Types:**
• **Structured:** "Top 10 movies of 2019"
• **Semantic:** "Movies about redemption"
• **Hybrid:** "Comedy movies with death themes"

**Tips:**
• Be specific with your criteria (year, genre, rating)
• Use natural language - I understand context!
• Ask follow-up questions to refine results

Need examples? Use /examples
    """
    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.MARKDOWN
    )


async def examples_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /examples command"""
    examples_text = """
📝 **Example Questions**

**Structured Queries:**
1. When did The Matrix release?
2. Top 5 movies of 2019 by meta score
3. Top 7 comedy movies between 2010-2020
4. Horror movies with meta score > 85 and IMDB > 8
5. Directors with movies grossing over $500M twice
6. Top 10 movies with over 1M votes

**Semantic Searches:**
7. Comedy movies with death themes
8. Summarize Steven Spielberg sci-fi movies
9. Movies before 1990 with police involvement

Try any of these or ask your own questions!
    """
    await update.message.reply_text(
        examples_text,
        parse_mode=ParseMode.MARKDOWN
    )


# =============================================================================
# MESSAGE HANDLER
# =============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages"""
    user_query = update.message.text
    user_name = update.effective_user.first_name or "there"

    # Validate API key
    if not OPENAI_API_KEY:
        await update.message.reply_text(
            "⚠️ Service configuration error. Please contact the administrator."
        )
        return

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    try:
        # Process query with agent
        result = query_agent(user_query)

        if result['success']:
            response = result['response']

            # Format response for Telegram (split if too long)
            max_length = 4096  # Telegram message limit

            if len(response) <= max_length:
                await update.message.reply_text(
                    response,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                # Split into chunks
                chunks = [response[i:i+max_length]
                          for i in range(0, len(response), max_length)]
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

            # Send query type info
            if result.get('query_type'):
                info_text = f"_Query Type: {result['query_type']}_"
                await update.message.reply_text(info_text, parse_mode=ParseMode.MARKDOWN)

        else:
            # Error response
            error_msg = f"⚠️ I encountered an error processing your query:\n{result.get('error', 'Unknown error')}"
            await update.message.reply_text(error_msg)

    except Exception as e:
        print(f"Error handling message: {e}")
        await update.message.reply_text(
            f"⚠️ Sorry, I encountered an unexpected error: {str(e)}\n\n"
            "Please try rephrasing your question or use /help for guidance."
        )


# =============================================================================
# ERROR HANDLER
# =============================================================================

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    print(f"Update {update} caused error {context.error}")

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ An error occurred while processing your request. Please try again."
        )


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run the bot"""
    # Validate configuration
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in environment variables")
        print("Please add TELEGRAM_BOT_TOKEN to your .env file")
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in environment variables")
        print("Please add OPENAI_API_KEY to your .env file")
        sys.exit(1)

    # Check if databases exist
    from pathlib import Path
    from config import DUCKDB_PATH, CHROMA_PATH

    if not DUCKDB_PATH.exists() or not CHROMA_PATH.exists():
        print("ERROR: Database setup incomplete!")
        print("Please run: python -m src.data_setup")
        sys.exit(1)

    print("Starting IMDB Telegram Bot...")
    print(f"Bot token: {TELEGRAM_BOT_TOKEN[:10]}...")

    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("examples", examples_command))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))

    # Add error handler
    application.add_error_handler(error_handler)

    # Start the bot
    print("Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
