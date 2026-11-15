"use client";

/**
 * MeetingQA — Ask a question about a completed MeetMind session.
 *
 * Calls POST /sessions/{id}/ask and reads the streaming SSE response,
 * rendering tokens as they arrive — same as a chat interface.
 *
 * Usage:
 *   <MeetingQA sessionId={session.id} token={token} />
 *
 * Drop this into the student or team session detail page below the
 * transcript/summary panels.
 */

import { useRef, useState } from "react";
import { Send, Bot, User, AlertCircle, Loader2 } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Message {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;   // true while tokens are still arriving
  error?: boolean;
}

interface Props {
  sessionId: string;
  token: string | null;
}

export function MeetingQA({ sessionId, token }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function scrollToBottom() {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  async function sendQuestion() {
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setLoading(true);

    // add user message immediately
    setMessages((prev) => [...prev, { role: "user", content: question }]);

    // add an empty assistant message that we'll fill with streaming tokens
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", streaming: true },
    ]);

    try {
      const res = await fetch(`${API_URL}/sessions/${sessionId}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        credentials: "include",   
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Request failed");
      }

      // read the SSE stream
      // Read the SSE stream
const reader = res.body?.getReader();
const decoder = new TextDecoder();

if (!reader) {
  throw new Error("No response body");
}

let buffer = "";

while (true) {
  const { done, value } = await reader.read();

  if (done) break;

  buffer += decoder.decode(value, { stream: true });

  const lines = buffer.split("\n");
  buffer = lines.pop() ?? "";

  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;

    const payload = line.slice(6);

    // Backend has finished sending the answer
    if (payload === "[DONE]") {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];

        if (last?.role === "assistant") {
          updated[updated.length - 1] = {
            ...last,
            streaming: false,
          };
        }

        return updated;
      });

      continue;
    }

    // Backend returned an error
    if (payload.startsWith("[ERROR]")) {
      throw new Error(payload.slice(8).trim());
    }

    // Append plain-text answer chunks
    setMessages((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];

      if (last?.role === "assistant") {
        updated[updated.length - 1] = {
          ...last,
          content: last.content + payload,
        };
      }

      return updated;
    });

    scrollToBottom();
  }
}
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last.role === "assistant") {
          updated[updated.length - 1] = {
            ...last,
            content: message,
            streaming: false,
            error: true,
          };
        }
        return updated;
      });
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
      scrollToBottom();
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // send on Enter, allow Shift+Enter for newline
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  }

  const suggestions = [
    "What were the main decisions made?",
    "What action items were assigned and to whom?",
    "What was the main blocker discussed?",
  ];

  return (
    <section
      className="rounded-card border border-border bg-surface"
      aria-label="Ask questions about this meeting"
    >
      {/* header */}
      <div className="flex items-center gap-2 border-b border-border px-5 py-4">
        <Bot size={16} className="text-signal" aria-hidden />
        <h2 className="font-display text-sm font-medium text-paper">
          Ask about this meeting
        </h2>
      </div>

      {/* message list */}
      <div className="flex max-h-[420px] flex-col gap-4 overflow-y-auto px-5 py-4">
        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="font-body text-xs text-muted">
              Ask anything about the transcript — decisions, action items, who
              said what, deadlines.
            </p>
            <div className="flex flex-wrap gap-2 pt-1">
              {suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="rounded-full border border-border bg-surface px-3 py-1 font-body text-xs text-muted transition hover:border-signal hover:text-paper"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
          >
            {/* avatar */}
            <div
              className={`mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full ${
                msg.role === "user" ? "bg-signal/20" : "bg-surface-2"
              }`}
              aria-hidden
            >
              {msg.role === "user" ? (
                <User size={12} className="text-signal" />
              ) : (
                <Bot size={12} className="text-muted" />
              )}
            </div>

            {/* bubble */}
            <div
              className={`max-w-[80%] rounded-card px-4 py-2.5 font-body text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-signal/10 text-paper"
                  : msg.error
                  ? "border border-red-500/30 bg-red-500/10 text-red-400"
                  : "border border-border bg-surface-2 text-paper"
              }`}
            >
              {msg.error && (
                <AlertCircle size={12} className="mb-1 inline text-red-400" />
              )}{" "}
              {msg.content || (msg.streaming ? null : "…")}
              {msg.streaming && (
                <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-signal" />
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* input */}
      <div className="border-t border-border px-5 py-4">
        <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the meeting…"
            rows={1}
            disabled={loading}
            className="flex-1 resize-none rounded-card border border-border bg-surface-2 px-4 py-2.5 font-body text-sm text-paper placeholder:text-muted focus:border-signal focus:outline-none disabled:opacity-50"
            style={{ minHeight: "42px", maxHeight: "120px" }}
            onInput={(e) => {
              // auto-grow textarea
              const t = e.currentTarget;
              t.style.height = "auto";
              t.style.height = `${t.scrollHeight}px`;
            }}
          />
          <button
            onClick={sendQuestion}
            disabled={loading || !input.trim()}
            aria-label="Send question"
            className="flex h-[42px] w-[42px] flex-shrink-0 items-center justify-center rounded-card bg-signal text-ink transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>
        <p className="mt-2 font-mono text-xs text-muted">
          Enter to send · Shift+Enter for newline · Answers grounded in this
          session's transcript only
        </p>
      </div>
    </section>
  );
}
