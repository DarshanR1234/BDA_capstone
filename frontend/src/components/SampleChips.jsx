import React from 'react';

const SAMPLES = [
  {
    id: 1,
    title: "CAA Bengal Violence",
    query: "क्या CAA विरोध प्रदर्शन के दौरान बंगाल में हिंदुओं को घर से बाहर निकालकर मारा गया?",
    tag: "FAKE",
    tagClass: "fake",
  },
  {
    id: 2,
    title: "Misbah-ul-Haq COVID",
    query: "वेस्टइंडीज दौरे पर कोरोना संक्रमित हुए पाकिस्तान के मुख्य कोच मिस्बाह-उल-हक",
    tag: "REAL",
    tagClass: "real",
  },
  {
    id: 3,
    title: "Modi Passing Rumor",
    query: "is narendra modi dead?",
    tag: "UNCERTAIN",
    tagClass: "uncertain",
  },
  {
    id: 4,
    title: "21-Day Lockdown Viral",
    query: "'अगले इक्कीस दिनों तक घरों से न निकलें,' मोदी ने कहा 21 दिनों का लॉकडाउन",
    tag: "FAKE",
    tagClass: "fake",
  },
];

export default function SampleChips({ onSelectSample, isLoading }) {
  return (
    <div className="chips-wrapper">
      <span className="chips-label">Try Benchmark Samples:</span>
      {SAMPLES.map((sample) => (
        <button
          key={sample.id}
          type="button"
          className="chip-btn"
          disabled={isLoading}
          onClick={() => onSelectSample(sample.query)}
        >
          <span>{sample.title}</span>
          <span className={`chip-tag ${sample.tagClass}`}>{sample.tag}</span>
        </button>
      ))}
    </div>
  );
}
