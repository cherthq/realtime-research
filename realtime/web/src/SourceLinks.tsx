export type WebSource = { url: string; title: string };

function readable(source: WebSource) {
  try {
    const url = new URL(source.url);
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) return null;
    return { ...source, host: url.hostname.replace(/^www\./, '') };
  } catch {
    return null;
  }
}

export function SourceLinks({ sources }: { sources?: WebSource[] }) {
  if (!Array.isArray(sources)) return null;
  const links = sources
    .slice(0, 5)
    .filter((s) => s && typeof s.url === 'string' && typeof s.title === 'string')
    .map(readable)
    .filter((s) => s !== null);
  if (!links.length) return null;
  return (
    <span className="source-links" aria-label="Answer sources">
      <span>Sources</span>
      {links.map((source, index) => (
        <a
          key={`${source.url}-${index}`}
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          title={source.title}
        >
          {index + 1}. {source.host}
        </a>
      ))}
    </span>
  );
}
