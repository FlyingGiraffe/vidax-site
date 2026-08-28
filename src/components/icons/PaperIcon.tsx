import React from 'react';

// Generic document/report glyph for the arXiv link -- not a reproduction of
// arXiv's own logo/wordmark, just a "paper" affordance icon.
export default function PaperIcon(props: React.SVGProps<SVGSVGElement>): React.ReactElement {
  return (
    <svg
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M3.5 1.5h6l3 3v10a.5.5 0 0 1-.5.5h-8.5a.5.5 0 0 1-.5-.5v-12a.5.5 0 0 1 .5-.5Z" />
      <path d="M9.5 1.5v3h3" />
      <path d="M5.25 8.25h5.5M5.25 10.25h5.5M5.25 12.25h3.5" />
    </svg>
  );
}
