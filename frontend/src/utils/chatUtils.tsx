import type { SourceChunk } from '../api/chat';

export const preprocessMessageContent = (
  text: string, 
  sources: SourceChunk[] = []
): string => {
  const blockRegex = /\[\s*source\s+\d+(?:\s*,\s*source\s+\d+)*\s*\]/gi;
  
  return text.replace(blockRegex, (match) => {
    const numRegex = /\d+/g;
    const nums = [];
    let numMatch;
    while ((numMatch = numRegex.exec(match)) !== null) {
      nums.push(parseInt(numMatch[0], 10));
    }
    
    const validLinks = [];
    const seenChunkIds = new Set<string>();
    
    for (const num of nums) {
      const source = sources.find(s => s.source_number === num);
      if (source && !seenChunkIds.has(source.chunk_id)) {
        // Output a special controlled markdown link using anchor to prevent stripping
        validLinks.push(`[Source ${num}](#source-${num})`);
        seenChunkIds.add(source.chunk_id);
      }
    }
    
    if (validLinks.length > 0) {
      return validLinks.join(' ');
    }
    
    // If all sources in this block were invalid, omit the marker completely.
    return '';
  });
};
