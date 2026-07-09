// Assistant-only dependency: other tabs do not pay for Markdown parsing
// or HTML sanitization.
import { marked } from "marked";
import DOMPurify from "dompurify";

export function markdown(text) {
  return DOMPurify.sanitize(marked.parse(String(text ?? "")));
}
