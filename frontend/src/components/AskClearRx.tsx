import { useState } from 'react';
import { apiService, type Citation } from '@/services/api';
import { Button } from '@/components/ui/button';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card';

export function AskClearRx() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [citations, setCitations] = useState<Citation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ask = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setAnswer('');
    setCitations([]);
    setError(null);
    await apiService.streamQuery(query.trim(), {
      onToken: (t) => setAnswer((a) => a + t),
      onCitations: (c) => setCitations(c),
      onDone: () => setLoading(false),
      onError: () => {
        setError('Sorry, the answer service is unavailable right now.');
        setLoading(false);
      },
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ask ClearRx</CardTitle>
        <CardDescription>
          Ask about a drug interaction in plain language. This is not medical advice.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <textarea
          className="w-full min-h-[80px] rounded-md border border-input bg-background p-2 text-sm"
          placeholder="e.g. Can I take ibuprofen with warfarin?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Button onClick={ask} disabled={loading || !query.trim()}>
          {loading ? 'Answering…' : 'Ask'}
        </Button>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {answer && <p className="whitespace-pre-wrap text-sm">{answer}</p>}
        {citations.length > 0 && (
          <div className="text-xs text-muted-foreground">
            <p className="font-medium">Sources</p>
            <ul className="list-disc pl-4">
              {citations.map((c, i) => (
                <li key={i}>
                  {c.url ? (
                    <a href={c.url} target="_blank" rel="noreferrer" className="underline">
                      {c.source_doc_id}
                    </a>
                  ) : (
                    c.source_doc_id
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
