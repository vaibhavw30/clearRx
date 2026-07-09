export type SSEEvent =
  | { type: 'token'; data: string }
  | { type: 'citations'; data: unknown }
  | { type: 'done' };

/**
 * Stateful parser for the ClearRx SSE stream. `feed` accepts an arbitrary
 * text chunk (frames may be split across chunks), buffers any incomplete
 * trailing frame, and returns the events completed so far.
 */
export function createSSEParser(): { feed(text: string): SSEEvent[] } {
  let buffer = '';
  return {
    feed(text: string): SSEEvent[] {
      buffer += text;
      const events: SSEEvent[] = [];
      let idx: number;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        if (!frame) continue;
        const lines = frame.split('\n');
        const isCitations = lines.some((l) => l.startsWith('event: citations'));
        const dataLine = lines.find((l) => l.startsWith('data: '));
        if (dataLine === undefined) continue;
        const payload = dataLine.slice('data: '.length);
        if (isCitations) {
          events.push({ type: 'citations', data: JSON.parse(payload) });
        } else if (payload === '[DONE]') {
          events.push({ type: 'done' });
        } else {
          events.push({ type: 'token', data: payload });
        }
      }
      return events;
    },
  };
}
