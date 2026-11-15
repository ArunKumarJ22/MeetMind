"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, MessageSquare } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

interface QAChatProps {
  sessionId: string;
  sessionStatus: string;
}

const SUGGESTED_QUESTIONS = [
  "What were the key decisions made?",
  "What action items were assigned and to whom?",
  "What was the main topic discussed?",
  "Were there any blockers or risks mentioned?",
];

export function QAChat({ sessionId, sessionStatus }: QAChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const isCompleted = sessionStatus === "completed";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendQuestion(question: string) {
    if (!question.trim() || isStreaming || !isCompleted) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question.trim(),
    };

    const assistantMsg: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsStreaming(true);

    try {
      const res = await fetch(`${API_URL}/sessions/${sessionId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ question: question.trim() }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Request failed");
      }

      // Read the SSE stream
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No response body");

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const token = line.slice(6); // strip "data: "
          if (token === "[DONE]") break;
          if (token.startsWith("[ERROR]")) {
            throw new Error(token.slice(8));
          }
          // append token to the streaming assistant message
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsg.id
                ? { ...m, content: m.content + token }
                : m
            )
          );
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Something went wrong";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id
            ? { ...m, content: `Sorry, I couldn't answer that: ${message}`, streaming: false }
            : m
        )
      );
    } finally {
      // mark streaming done
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id ? { ...m, streaming: false } : m
        )
      );
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    sendQuestion(input);
  }

  if (!isCompleted) {
    return (
      <div className="rounded-xl border border-border bg-surface p-6 text-center">
        <MessageSquare className="mx-auto mb-3 h-8 w-8 text-muted" />
        <p className="font-body text-sm text-muted">
          Q&amp;A will be available once the session finishes processing.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col rounded-xl border border-border bg-surface">
      {/* header */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <MessageSquare className="h-4 w-4 text-accent" />
        <span className="font-body text-sm font-medium text-paper">
          Ask about this meeting
        </span>
      </div>

      {/* messages */}
      <div className="flex min-h-[220px] flex-col gap-3 overflow-y-auto p-4" style={{ maxHeight: 400 }}>
        {messages.length === 0 && (
          <div className="mt-2">
            <p className="mb-3 font-body text-xs text-muted">Try asking:</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => sendQuestion(q)}
                  className="rounded-full border border-border px-3 py-1 font-body text-xs text-muted transition hover:border-accent hover:text-accent"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-xl px-4 py-2.5 font-body text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-accent text-white"
                  : "bg-surface-2 text-paper"
              }`}
            >
              {msg.content || (msg.streaming && (
                <span className="inline-flex items-center gap-1 text-muted">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Thinking…
                </span>
              ))}
              {msg.streaming && msg.content && (
                <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-current opacity-70" />
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* input */}
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2 border-t border-border p-3"
      >
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isStreaming}
          placeholder="Ask anything about this meeting…"
          className="flex-1 rounded-lg border border-border bg-background px-3 py-2 font-body text-sm text-paper placeholder:text-muted focus:border-accent focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isStreaming || !input.trim()}
          className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-white transition hover:opacity-90 disabled:opacity-40"
          aria-label="Send question"
        >
          {isStreaming ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </button>
      </form>
    </div>
  );
}
