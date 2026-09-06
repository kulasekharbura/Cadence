import { describe, it, expect } from 'vitest';
import { preprocessMessageContent } from './chatUtils';
import type { SourceChunk } from '../api/chat';

describe('chatUtils preprocessMessageContent', () => {
  const mockSources: SourceChunk[] = [
    {
      source_number: 1,
      chunk_id: 'chunk-1',
      document_id: 'doc-1',
      filename: 'Resume_kulasekhar.pdf',
      page_number: 1,
      chunk_index: 0
    },
    {
      source_number: 2,
      chunk_id: 'chunk-2',
      document_id: 'doc-1',
      filename: 'Assessment_Guidelines.pdf',
      page_number: 1,
      chunk_index: 0
    }
  ];

  it('renders one markdown link for [Source 1]', () => {
    const text = 'Here is the answer [Source 1].';
    const result = preprocessMessageContent(text, mockSources);
    expect(result).toBe('Here is the answer [Source 1](#source-1).');
  });

  it('renders one markdown link for [Source 2]', () => {
    const text = 'Here is the answer [Source 2].';
    const result = preprocessMessageContent(text, mockSources);
    expect(result).toBe('Here is the answer [Source 2](#source-2).');
  });

  it('renders two markdown links for grouped [Source 1, Source 2]', () => {
    const text = 'Answer [Source 1, Source 2]';
    const result = preprocessMessageContent(text, mockSources);
    expect(result).toBe('Answer [Source 1](#source-1) [Source 2](#source-2)');
  });

  it('renders Source 1 link and omits invalid Source 99 in grouped [Source 1, Source 99]', () => {
    const text = 'Answer [Source 1, Source 99]';
    const result = preprocessMessageContent(text, mockSources);
    expect(result).toBe('Answer [Source 1](#source-1)');
  });

  it('omits citation completely for invalid [Source 99]', () => {
    const text = 'Answer [Source 99] here.';
    const result = preprocessMessageContent(text, mockSources);
    // Note: Because it omits the marker entirely, there will be two spaces.
    expect(result).toBe('Answer  here.');
  });

  it('renders two links for consecutive [Source 1][Source 2]', () => {
    const text = 'Answer [Source 1][Source 2]';
    const result = preprocessMessageContent(text, mockSources);
    expect(result).toBe('Answer [Source 1](#source-1)[Source 2](#source-2)');
  });

  it('handles markdown text surrounding the citation', () => {
    const text = '**B.Tech** [Source 1]';
    const result = preprocessMessageContent(text, mockSources);
    expect(result).toBe('**B.Tech** [Source 1](#source-1)');
  });
});
