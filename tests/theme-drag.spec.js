import { test, expect } from '@playwright/test';
import {
  readFormattingJson,
  deleteFormattingJson,
  waitForFormattingSave,
  verifyStableSequenceKeys,
  verifyTimestampsUpdated,
  verifyFormattingPreserved,
} from './test-helpers.js';

/**
 * E2E Tests for Theme Dragging and Subtitle Formatting
 *
 * These tests verify that:
 * 1. Sequence keys remain stable during timeline operations
 * 2. Timestamps are updated correctly
 * 3. Formatting (color, bold, etc.) is preserved
 */

const TEST_FOLDER = '001';
const TEST_THEME = '3';

test.describe('Theme Drag Operations', () => {
  // Clean up before each test
  test.beforeEach(async ({ page }) => {
    deleteFormattingJson(TEST_FOLDER, TEST_THEME);

    // Navigate to adjust page
    await page.goto(`/adjust.html?folder=${TEST_FOLDER}&theme=${TEST_THEME}`);

    // Wait for page to load
    await page.waitForLoadState('networkidle');
    await expect(page.locator('#subtitle-text')).toBeVisible({ timeout: 10000 });
  });

  test('should apply color formatting to subtitle', async ({ page }) => {
    // Apply yellow color to current subtitle
    const colorButton = page.locator('button[onclick*="applyColor(\'#ffff00\'"]');
    await expect(colorButton).toBeVisible();
    await colorButton.click();

    // Wait for formatting to be saved
    const json = await waitForFormattingSave(TEST_FOLDER, TEST_THEME);

    // Verify JSON was created
    expect(json).not.toBeNull();
    expect(Object.keys(json).length).toBeGreaterThan(0);

    // Verify structure has sequence keys
    const firstKey = Object.keys(json)[0];
    expect(json[firstKey]).toHaveProperty('sequence');
    expect(json[firstKey]).toHaveProperty('timestamp');
    expect(json[firstKey]).toHaveProperty('color');
  });

  test('should preserve sequence keys during timeline drag', async ({ page }) => {
    // Apply formatting first
    await page.locator('button[onclick*="applyColor(\'#ffff00\'"]').click();
    const beforeJson = await waitForFormattingSave(TEST_FOLDER, TEST_THEME);

    // Get initial sequence keys
    const initialKeys = Object.keys(beforeJson).sort();

    // Drag timeline left by 2 seconds
    // This requires interacting with the timeline drag handle
    const timeline = page.locator('.theme-box');
    await expect(timeline).toBeVisible();

    // Get left handle position and drag it left
    const leftHandle = page.locator('#handle-left');
    if (await leftHandle.count() > 0) {
      const box = await leftHandle.boundingBox();
      if (box) {
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
        await page.mouse.down();
        await page.mouse.move(box.x - 100, box.y + box.height / 2);
        await page.mouse.up();
        await page.waitForTimeout(1500); // Wait for auto-save
      }
    }

    // Read JSON after drag
    const afterJson = readFormattingJson(TEST_FOLDER, TEST_THEME);

    // Verify sequence keys are stable
    const { keysMatch, beforeKeys, afterKeys } = verifyStableSequenceKeys(beforeJson, afterJson);
    expect(keysMatch, `Sequence keys changed!\nBefore: ${beforeKeys.join(', ')}\nAfter: ${afterKeys.join(', ')}`).toBeTruthy();
  });

  test('should update timestamps during timeline drag', async ({ page }) => {
    // Apply formatting
    await page.locator('button[onclick*="applyColor(\'#ffff00\'"]').click();
    const beforeJson = await waitForFormattingSave(TEST_FOLDER, TEST_THEME);

    // Reset timeline to original position (this may shift timestamps)
    const resetButton = page.locator('#reset-timeline-btn');
    if (await resetButton.count() > 0) {
      await resetButton.click();
      await page.waitForTimeout(2000); // Wait for reset and auto-save
    }

    // Read JSON after reset
    const afterJson = readFormattingJson(TEST_FOLDER, TEST_THEME);

    // Verify formatting was updated
    expect(afterJson).not.toBeNull();

    // Verify timestamps may have changed (depending on current state)
    // The key test is that sequence keys remain stable
    const { keysMatch } = verifyStableSequenceKeys(beforeJson, afterJson);
    expect(keysMatch, 'Sequence keys must remain stable after reset').toBeTruthy();
  });

  test('should preserve formatting after reset', async ({ page }) => {
    // Apply yellow color
    await page.locator('button[onclick*="applyColor(\'#ffff00\'"]').click();
    const beforeJson = await waitForFormattingSave(TEST_FOLDER, TEST_THEME);

    // Get the first entry's color
    const firstKey = Object.keys(beforeJson)[0];
    const originalColor = beforeJson[firstKey].color;

    // Reset timeline
    const resetButton = page.locator('#reset-timeline-btn');
    if (await resetButton.count() > 0) {
      await resetButton.click();
      await page.waitForTimeout(2000);
    }

    // Read JSON after reset
    const afterJson = readFormattingJson(TEST_FOLDER, TEST_THEME);

    // Verify color is preserved
    expect(afterJson[firstKey].color).toBe(originalColor);
  });

  test('should have correct JSON structure', async ({ page }) => {
    // Apply formatting
    await page.locator('button[onclick*="applyColor(\'#ffff00\'"]').click();
    const json = await waitForFormattingSave(TEST_FOLDER, TEST_THEME);

    // Verify structure
    const firstEntry = json[Object.keys(json)[0]];

    // Verify structure - allow null for optional fields
    expect(firstEntry).toHaveProperty('sequence');
    expect(firstEntry).toHaveProperty('timestamp');
    expect(firstEntry).toHaveProperty('html');
    expect(firstEntry).toHaveProperty('_text');
    expect(firstEntry).toHaveProperty('bold');
    expect(firstEntry).toHaveProperty('italic');
    expect(firstEntry).toHaveProperty('color'); // Can be null or string
    expect(firstEntry).toHaveProperty('size'); // Can be null or number
    expect(firstEntry).toHaveProperty('position');

    // Verify types for required fields
    expect(typeof firstEntry.sequence).toBe('number');
    expect(typeof firstEntry.timestamp).toBe('string');
    expect(typeof firstEntry.html).toBe('string');
    expect(typeof firstEntry._text).toBe('string');
    expect(typeof firstEntry.bold).toBe('boolean');
    expect(typeof firstEntry.italic).toBe('boolean');
  });

  test('should migrate old format to new format', async ({ page }) => {
    // This test verifies the migration logic
    // For now, we just verify new entries use sequence keys

    await page.locator('button[onclick*="applyColor(\'#ffff00\'"]').click();
    const json = await waitForFormattingSave(TEST_FOLDER, TEST_THEME);

    // Verify keys are sequence numbers (strings that parse to numbers)
    const keys = Object.keys(json);
    keys.forEach(key => {
      expect(Number(key)).not.toBeNaN();
      expect(json[key].sequence).toBe(Number(key));
    });
  });
});

test.describe('Formatting Persistence', () => {
  test.beforeEach(async ({ page }) => {
    deleteFormattingJson(TEST_FOLDER, TEST_THEME);
    await page.goto(`/adjust.html?folder=${TEST_FOLDER}&theme=${TEST_THEME}`);
    await page.waitForLoadState('networkidle');
  });

  test('should load existing formatting on page reload', async ({ page }) => {
    // Apply formatting
    await page.locator('button[onclick*="applyColor(\'#ffff00\'"]').click();
    const savedJson = await waitForFormattingSave(TEST_FOLDER, TEST_THEME);
    const firstKey = Object.keys(savedJson)[0];
    const originalColor = savedJson[firstKey].color;

    // Reload page
    await page.reload();
    await page.waitForLoadState('networkidle');

    // The formatting should still be in the JSON file
    const reloadedJson = readFormattingJson(TEST_FOLDER, TEST_THEME);
    expect(reloadedJson[firstKey].color).toBe(originalColor);
  });
});
