"use client";

import { useRef } from "react";
import { useLiveSession } from "../../lib/useLiveSession";
import { SummarySections } from "../../components/SummarySections";
import { QuizPanel } from "../../components/QuizPanel";
import { StatusBadge } from "../../components/StatusBadge";
import { WaveformMark } from "../../components/WaveformMark";
import { ProcessingStages } from "../../components/ProcessingStages";
import { TranscriptView } from "../../components/TranscriptView";
import { AudioPlayer, AudioPlayerHandle } from "../../components/AudioPlayer";
import { api } from "../../lib/api";
import { MeetingQA } from "../../components/MeetingQA";
import { useAuth } from "../../lib/auth-context";

export default function StudentSessionPage({ params }: { params: { sessionId: string } }) {
  const { token } = useAuth();
  const { session, stage, error } = useLiveSession(params.sessionId);
  const playerRef = useRef<AudioPlayerHandle>(null);

  function seekTo(seconds: number) {
    playerRef.current?.seekTo(seconds);
  }

  if (error) {
    return <div className="mx-auto max-w-3xl px-6 py-16 font-body text-red-400">{error}</div>;
  }

  if (!session) {
    return <div className="mx-auto max-w-3xl px-6 py-16 font-body text-muted">Loading…</div>;
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="font-display text-2xl font-medium text-paper">{session.title}</h1>
          <p className="font-mono text-xs text-muted">
            {session.processing_seconds != null && `Processed in ${session.processing_seconds}s`}
          </p>
        </div>
        <StatusBadge status={session.status} />
      </div>

      {session.status !== "completed" && session.status !== "failed" && (
        <div className="flex flex-col items-center gap-6 rounded-card border border-border bg-surface py-16 text-center">
          <WaveformMark className="h-10 w-64 opacity-70" />
          <ProcessingStages currentStage={stage} />
        </div>
      )}

      {session.status === "failed" && (
        <div className="rounded-card border border-red-500/30 bg-red-500/10 p-5 font-body text-sm text-red-300">
          Processing failed: {session.error_message}
        </div>
      )}

      {session.status === "completed" && session.student_output && (
        <div className="space-y-10">
          <section>
            <AudioPlayer ref={playerRef} src={api.audioUrl(session.id)} />
          </section>

          <section>
            <h2 className="mb-4 font-display text-lg font-medium text-paper">Topic summary</h2>
            <SummarySections sections={session.student_output.sections} />
          </section>

          <section>
            <h2 className="mb-4 font-display text-lg font-medium text-paper">Quiz yourself</h2>
            <QuizPanel
              quiz={session.student_output.quiz}
              flaggedCount={session.student_output.hallucination_flagged_count}
              onSeek={seekTo}
            />
          </section>

          {session.transcript && (
            <section>
              <h2 className="mb-4 font-display text-lg font-medium text-paper">Full transcript</h2>
              <TranscriptView segments={session.transcript.segments} onSeek={seekTo} />
            </section>
          )}
          {session.status === "completed" && (
            <MeetingQA sessionId={session.id} token={token} />
          )}
        </div>
      )}

    </div>
  );
}
