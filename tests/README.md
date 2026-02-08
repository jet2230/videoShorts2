# E2E Tests for videoShorts2

End-to-end tests using Playwright to verify theme dragging and subtitle formatting functionality.

## Prerequisites

- Node.js and npm installed
- Python 3 for the server
- Test data in `media/001_videos/` folder

## Running Tests

### Run all tests (headless)
```bash
npm test
```

### Run tests in UI mode (recommended)
```bash
npm run test:ui
```

This opens the Playwright UI where you can:
- See tests running in real-time
- Inspect DOM elements
- View network requests
- Debug failures

### Run tests in headed mode (visible browser)
```bash
npm run test:headed
```

### Debug tests
```bash
npm run test:debug
```

This pauses execution and lets you step through tests.

### View test report
```bash
npm run test:report
```

## Test Structure

```
tests/
├── theme-drag.spec.js    # Main test file for theme dragging
├── test-helpers.js       # Utility functions for reading JSON, verifying state
└── README.md            # This file
```

## Test Coverage

### Theme Drag Operations
- ✅ Apply color formatting to subtitle
- ✅ Preserve sequence keys during timeline drag
- ✅ Update timestamps during timeline drag
- ✅ Preserve formatting after reset
- ✅ Verify correct JSON structure
- ✅ Migrate old format to new format

### Formatting Persistence
- ✅ Load existing formatting on page reload

## What the Tests Verify

1. **Sequence Key Stability**: JSON keys (sequence numbers) never change during timeline operations
2. **Timestamp Updates**: Timestamps are correctly updated when timeline is dragged/reset
3. **Formatting Preservation**: Colors, bold, italic, etc. are preserved across operations
4. **JSON Structure**: New format has correct structure with `sequence`, `timestamp`, `html`, etc.

## Test Data

Tests use:
- **Folder**: 001
- **Theme**: 3
- **Formatting File**: `media/001_videos/shorts/theme_003_formatting.json`

**Note**: Tests clean up (delete) the formatting JSON file before each test run.

## Troubleshooting

### Tests fail with "Server not running"
Make sure the server starts on port 5000. Playwright will automatically start it using `python3 server.py`.

### Tests fail with "Test data not found"
Ensure you have:
- `media/001_videos/` folder
- `media/001_videos/shorts/` folder
- Theme 3 data available

### Tests timeout
Increase timeout in `playwright.config.js`:
```js
use: {
  actionTimeout: 10000,  // Increase from default
}
```

### Debugging a specific test
```bash
npx playwright test -g "test name" --debug
```

## Adding New Tests

1. Add test to `theme-drag.spec.js` or create new `.spec.js` file
2. Use helpers from `test-helpers.js`:
   - `readFormattingJson(folder, theme)`
   - `waitForFormattingSave(folder, theme)`
   - `verifyStableSequenceKeys(before, after)`
   - `verifyTimestampsUpdated(before, after, offset)`
   - `verifyFormattingPreserved(before, after, keys)`

Example:
```javascript
test('my new test', async ({ page }) => {
  await page.goto('/adjust.html?folder=001&theme=3');
  await page.waitForLoadState('networkidle');

  // Do something
  await page.click('#some-button');

  // Verify result
  const json = await waitForFormattingSave('001', '3');
  expect(json).not.toBeNull();
});
```
