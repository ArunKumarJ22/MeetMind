const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type SessionType = "student" | "professional";
export type SessionStatus = "live" | "uploaded" | "processing" | "completed" | "failed";

export interface UserOut {
  id: string;
  email: string;
  full_name: string;
  account_type: SessionType;
}

export interface TokenOut {
  access_token: string;
  token_type: string;
  user: UserOut;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  speaker?: string | null;
}

export interface ConceptSection {
  title: string;
  start: number;
  end: number;
  summary: string;
  key_concepts: string[];
}

export interface QuizQuestion {
  bloom_level: "remembering" | "understanding" | "applying";
  question: string;
  options: string[];
  correct_index: number;
  source_segment_start: number;
  source_segment_end: number;
  evidence_quote: string;
}

export interface ActionItem {
  owner: string;
  task: string;
  due?: string | null;
  source_segment_start: number;
  source_segment_end: number;
  confidence: number;
}

export interface RoleSummary {
  role: string;
  summary: string;
}

export interface RetrievedDocument {
  doc_id: string;
  title: string;
  score: number;
  triggered_by_quote: string;
}

export interface SessionOut {
  id: string;
  title: string;
  session_type: SessionType;
  status: SessionStatus;
  error_message?: string | null;
  processing_seconds?: number | null;
  transcript?: { segments: TranscriptSegment[]; raw_text: string } | null;
  student_output?: { sections: ConceptSection[]; quiz: QuizQuestion[]; hallucination_flagged_count: number } | null;
  professional_output?: { action_items: ActionItem[]; role_summaries: RoleSummary[]; action_item_bertscore: number | null } | null;
  retrieved_documents?: RetrievedDocument[] | null;
  created_at: string;
  completed_at?: string | null;
}

export interface SessionListItem {
  id: string;
  title: string;
  session_type: SessionType;
  status: SessionStatus;
  created_at: string;
}

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}, token?: string | null): Promise<T> {
  const headers: Record<string, string> = { ...(options.headers as Record<string, string>) };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
    // Always send the httpOnly auth cookie (set by /auth/login and /auth/register)
    // — this is the primary auth path now; the Authorization header above is kept
    // for backward compatibility / non-browser API clients, not required for the
    // web app to function.
    credentials: "include",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* response wasn't JSON */
    }
    throw new ApiError(detail, res.status);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  register: (email: string, password: string, full_name: string, account_type: SessionType) =>
    request<TokenOut>("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name, account_type }),
    }),

  login: (email: string, password: string) =>
    request<TokenOut>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),

  logout: () => request<void>("/auth/logout", { method: "POST" }),

  me: (token?: string | null) => request<UserOut>("/auth/me", {}, token),

  listSessions: (token?: string | null) => request<SessionListItem[]>("/sessions", {}, token),

  getSession: (token: string | null | undefined, sessionId: string) =>
    request<SessionOut>(`/sessions/${sessionId}`, {}, token),

  createSession: (token: string | null | undefined, title: string, sessionType: SessionType, audio: File) => {
    const form = new FormData();
    form.append("title", title);
    form.append("session_type", sessionType);
    form.append("audio", audio);
    return request<SessionOut>("/sessions", { method: "POST", body: form }, token);
  },

  createLiveSession: (token: string | null | undefined, title: string, sessionType: SessionType) =>
    request<SessionOut>("/sessions/live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, session_type: sessionType }),
    }, token),

  attachAudio: (token: string | null | undefined, sessionId: string, audio: Blob, filename: string) => {
    const form = new FormData();
    form.append("audio", audio, filename);
    return request<SessionOut>(`/sessions/${sessionId}/audio`, { method: "POST", body: form }, token);
  },

  deleteSession: (token: string | null | undefined, sessionId: string) =>
    request<void>(`/sessions/${sessionId}`, { method: "DELETE" }, token),

  uploadDocument: (token: string | null | undefined, sessionId: string, title: string, text: string) =>
    request(`/sessions/${sessionId}/documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text }),
    }, token),

  audioUrl: (sessionId: string) => `${API_URL}/sessions/${sessionId}/audio`,

  wsUrl: (sessionId: string) => {
    const wsBase = API_URL.replace(/^http/, "ws");
    return `${wsBase}/ws/sessions/${sessionId}`;
  },

  roomWsUrl: (roomId: string, userId: string, name: string) => {
    const wsBase = API_URL.replace(/^http/, "ws");
    const params = new URLSearchParams({ user_id: userId, name });
    return `${wsBase}/ws/rooms/${roomId}?${params.toString()}`;
  },
  async askSession(
  token: string | null,
  sessionId: string,
  question: string,
): Promise<Response> {
  // Returns the raw Response so the caller can read the SSE stream.
  // The MeetingQA component handles the streaming itself.
  return fetch(`${API_URL}/sessions/${sessionId}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ question }),
  });
},
};

export { ApiError };
