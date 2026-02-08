import { readFileSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';

/**
 * Test helper utilities for videoShorts2 E2E tests
 */

/**
 * Find the actual folder name for a given folder number
 * (e.g., "001" -> "001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6")
 */
function findFolderByName(folderNumber, baseDir = './videos') {
  try {
    const folders = readdirSync(baseDir);
    const folder = folders.find(f => f.startsWith(`${folderNumber}_`));
    return folder || null;
  } catch (error) {
    return null;
  }
}

/**
 * Get the path to the formatting JSON file for a theme
 */
export function getFormattingFilePath(folderNumber, themeNumber) {
  const baseDir = process.env.VIDEO_OUTPUT_DIR || './videos';
  const folderName = findFolderByName(folderNumber, baseDir);

  if (!folderName) {
    throw new Error(`Folder not found for folder number ${folderNumber}`);
  }

  const themeNum = parseInt(themeNumber);
  const themePadded = themeNum.toString().padStart(3, '0');
  return join(
    baseDir,
    folderName,
    'shorts',
    `theme_${themePadded}_formatting.json`
  );
}

/**
 * Read the formatting JSON file
 */
export function readFormattingJson(folderNumber, themeNumber) {
  const filePath = getFormattingFilePath(folderNumber, themeNumber);

  if (!existsSync(filePath)) {
    return null;
  }

  try {
    const content = readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    console.error(`Failed to read formatting JSON: ${error.message}`);
    return null;
  }
}

/**
 * Delete formatting JSON file (for test cleanup)
 */
export function deleteFormattingJson(folderNumber, themeNumber) {
  const { unlinkSync } = require('fs');
  const filePath = getFormattingFilePath(folderNumber, themeNumber);

  if (existsSync(filePath)) {
    try {
      unlinkSync(filePath);
      return true;
    } catch (error) {
      console.error(`Failed to delete formatting JSON: ${error.message}`);
      return false;
    }
  }
  return false;
}

/**
 * Wait for formatting JSON to be saved (with timeout)
 */
export async function waitForFormattingSave(folderNumber, themeNumber, timeout = 5000) {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const json = readFormattingJson(folderNumber, themeNumber);
    if (json && Object.keys(json).length > 0) {
      return json;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  throw new Error(`Formatting JSON not saved within ${timeout}ms`);
}

/**
 * Verify sequence keys are stable (no regeneration)
 */
export function verifyStableSequenceKeys(beforeJson, afterJson) {
  const beforeKeys = Object.keys(beforeJson).sort();
  const afterKeys = Object.keys(afterJson).sort();

  return {
    keysMatch: JSON.stringify(beforeKeys) === JSON.stringify(afterKeys),
    beforeKeys,
    afterKeys,
  };
}

/**
 * Verify timestamps were updated by a specific offset
 */
export function verifyTimestampsUpdated(beforeJson, afterJson, expectedOffsetSeconds) {
  const vttToSeconds = (vtt) => {
    const [time, millis] = vtt.split('.');
    const isNegative = time.startsWith('-');
    const [h, m, s] = time.replace('-', '').split(':').map(Number);
    const result = h * 3600 + m * 60 + s + (millis || 0) / 1000;
    return isNegative ? -result : result;
  };

  const updates = [];

  for (const [key, beforeEntry] of Object.entries(beforeJson)) {
    if (!afterJson[key]) {
      updates.push({ key, status: 'missing', error: 'Key not found in after JSON' });
      continue;
    }

    const beforeSeconds = vttToSeconds(beforeEntry.timestamp);
    const afterSeconds = vttToSeconds(afterJson[key].timestamp);
    const actualOffset = afterSeconds - beforeSeconds;
    const offsetDiff = Math.abs(actualOffset - expectedOffsetSeconds);

    updates.push({
      key,
      status: offsetDiff < 0.1 ? 'ok' : 'mismatch',
      beforeTimestamp: beforeEntry.timestamp,
      afterTimestamp: afterJson[key].timestamp,
      expectedOffset: expectedOffsetSeconds,
      actualOffset,
    });
  }

  return updates;
}

/**
 * Verify formatting is preserved
 */
export function verifyFormattingPreserved(beforeJson, afterJson, keys = null) {
  const keysToCheck = keys || Object.keys(beforeJson);
  const results = [];

  for (const key of keysToCheck) {
    const before = beforeJson[key];
    const after = afterJson[key];

    if (!before || !after) {
      results.push({ key, status: 'missing', error: 'Entry missing' });
      continue;
    }

    const preserved = {
      color: before.color === after.color,
      bold: before.bold === after.bold,
      italic: before.italic === after.italic,
      size: before.size === after.size,
      position: before.position === after.position,
    };

    const allPreserved = Object.values(preserved).every(v => v === true);

    results.push({
      key,
      status: allPreserved ? 'preserved' : 'changed',
      preserved,
      before: { color: before.color, bold: before.bold, italic: before.italic },
      after: { color: after.color, bold: after.bold, italic: after.italic },
    });
  }

  return results;
}
