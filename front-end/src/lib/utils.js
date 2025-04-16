/**
 * Combines multiple class names into a single string, filtering out falsy values
 * @param  {...any} classes - Class names to combine
 * @returns {string} - Combined class names
 */
export function cn(...classes) {
  return classes.filter(Boolean).join(' ');
} 