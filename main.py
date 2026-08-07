import sys
import db
from engine import GeminiEngine

SYSTEM_PROMPT = """
You are NOVA, an intelligent, warm, and highly capable personal AI assistant.

Core Persona & Guidelines:
1. Friendly & Grounded: Speak naturally, warmly, and encouragingly.
2. Structured & Clear: Deliver answers with maximum clarity. Use concise bullet points and bold key terms.
3. Solution-Oriented: Focus directly on resolving the user's intent immediately.
4. Actionable & Helpful: Answer the core intent first, then offer next steps.
"""

def main():
    print("==================================================")
    print("  NOVA AI Engine (Persistent SQLite Storage)     ")
    print("  Commands: '/clear' (reset DB), 'exit' (quit)    ")
    print("==================================================\n")

    engine = GeminiEngine(system_instruction=SYSTEM_PROMPT)

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("Terminating engine session.")
                break
                
            if user_input.lower() == "/clear":
                db.clear_history()
                engine.history = []
                print("\n[SYSTEM]: SQLite database history cleared successfully.\n")
                continue

            print("\nNOVA > ", end="", flush=True)

            for chunk in engine.stream_message(user_input):
                sys.stdout.write(chunk)
                sys.stdout.flush()

            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nSession forcibly closed.")
            break

if __name__ == "__main__":
    main()