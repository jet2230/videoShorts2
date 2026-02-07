#!/usr/bin/env python3
"""
Test suite for subtitle HTML → ASS conversion
Tests the ass_formatter.py module with various styling scenarios
"""

from ass_formatter import ASSFormatter
import json

# Test cases
test_cases = [
    # 1. Single Style Tests
    {
        "id": "TC1",
        "name": "Single word - Bold only",
        "input": 'word <b>test</b> word',
        "expected_contains": ["{\\b1}test{\\b0}"]
    },
    {
        "id": "TC2",
        "name": "Single word - Italic only",
        "input": 'word <i>test</i> word',
        "expected_contains": ["{\\i1}test{\\i0}"]
    },
    {
        "id": "TC3",
        "name": "Single word - Color only (yellow)",
        "input": '<span style="color: #ffff00;">test</span>',
        "expected_contains": ["{\\c&H0000FFFF}test", "{\\c&H00FFFFFF}"]
    },
    {
        "id": "TC4",
        "name": "Single word - Font size only (larger)",
        "input": '<span style="font-size: 1.6em;">test</span>',
        "expected_contains": ["{\\fs77}test", "{\\r}"]
    },

    # 2. Combined Style Tests
    {
        "id": "TC5",
        "name": "Bold + Italic",
        "input": '<b><i>test</i></b>',
        "expected_contains": ["{\\b1}", "{\\i1}test", "{\\i0}", "{\\b0}"]
    },
    {
        "id": "TC6",
        "name": "Bold + Color (cyan)",
        "input": '<font color="#00ffff"><b>people</b></font>',
        "expected_contains": ["{\\c&H00FFFF00}", "{\\b1}people", "{\\b0}"]
    },
    {
        "id": "TC7",
        "name": "Italic + Color (yellow)",
        "input": '<span style="color: #ffff00;"><i>test</i></span>',
        "expected_contains": ["{\\c&H0000FFFF}", "{\\i1}test"]
    },
    {
        "id": "TC8",
        "name": "Bold + Italic + Color (red)",
        "input": '<font color="#ff0000"><b><i>test</i></b></font>',
        "expected_contains": ["{\\c&H000000FF}", "{\\b1}", "{\\i1}test"]
    },
    {
        "id": "TC9",
        "name": "Color + Size (combined font tag)",
        "input": '<font color="#00ffff" size="6">people</font>',
        "expected_contains": ["{\\c&H00FFFF00}", "{\\fs77}people"]
    },
    {
        "id": "TC10",
        "name": "Bold + Color + Size (all three)",
        "input": '<font color="#00ffff" size="6"><b>people</b></font>',
        "expected_contains": ["{\\c&H00FFFF00}", "{\\fs77}", "{\\b1}people"]
    },

    # 3. Multiple Words with Same Style
    {
        "id": "TC11",
        "name": "Bold phrase",
        "input": 'This is <b>a very important</b> message',
        "expected_contains": ["{\\b1}a very important{\\b0}"]
    },
    {
        "id": "TC12",
        "name": "Colored phrase (yellow)",
        "input": '<span style="color: #ffff00;">notice this text</span> here',
        "expected_contains": ["{\\c&H0000FFFF}notice this text"]
    },
    {
        "id": "TC13",
        "name": "Large text phrase",
        "input": '<span style="font-size: 1.8em;">big text here</span>',
        "expected_contains": ["{\\fs86}big text here"]
    },

    # 4. Mixed/Different Styles in One Line
    {
        "id": "TC14",
        "name": "Two different colors",
        "input": '<span style="color: #ffff00;">yellow</span> and <span style="color: #00ffff;">cyan</span>',
        "expected_contains": ["{\\c&H0000FFFF}yellow", "{\\c&H00FFFF00}cyan"]
    },
    {
        "id": "TC15",
        "name": "Bold and italic in same line",
        "input": '<b>bold text</b> and <i>italic text</i>',
        "expected_contains": ["{\\b1}bold text{\\b0}", "{\\i1}italic text{\\i0}"]
    },
    {
        "id": "TC16",
        "name": "Three different styled sections",
        "input": '<b>bold</b>, <i>italic</i>, and <span style="color: #ff0000;">red</span>',
        "expected_contains": ["{\\b1}bold", "{\\i1}italic", "{\\c&H000000FF}red"]
    },
    {
        "id": "TC17",
        "name": "Complex mix with all styles",
        "input": '<b>Bold</b> normal <i>italic</i> normal <span style="color: #ffff00;">yellow</span> normal <font color="#00ffff" size="6"><b>big cyan</b></font>',
        "expected_contains": ["{\\b1}Bold", "{\\i1}italic", "{\\c&H0000FFFF}yellow", "{\\c&H00FFFF00}", "{\\fs77}", "{\\b1}big cyan"]
    },

    # 5. Nested/Overlapping Styles
    {
        "id": "TC18",
        "name": "Bold inside colored",
        "input": '<span style="color: #ffff00;"><b>yellow and bold</b></span>',
        "expected_contains": ["{\\c&H0000FFFF}", "{\\b1}yellow and bold"]
    },
    {
        "id": "TC19",
        "name": "Colored inside bold",
        "input": '<b><span style="color: #00ffff;">cyan and bold</span></b>',
        "expected_contains": ["{\\b1}", "{\\c&H00FFFF00}cyan and bold"]
    },
    {
        "id": "TC20",
        "name": "Size + bold + color nested",
        "input": '<font color="#ff0000" size="5"><b><i>all styles</i></b></font>',
        "expected_contains": ["{\\c&H000000FF}", "{\\fs67}", "{\\b1}", "{\\i1}all styles"]
    },

    # 6. Font Size Variations
    {
        "id": "TC21",
        "name": "Size 1 (smallest - 0.6em)",
        "input": '<span style="font-size: 0.6em;">tiny</span>',
        "expected_contains": ["{\\fs29}tiny"]
    },
    {
        "id": "TC22",
        "name": "Size 4 (normal - 1.2em)",
        "input": '<span style="font-size: 1.2em;">normal</span>',
        "expected_contains": ["{\\fs58}normal"]
    },
    {
        "id": "TC23",
        "name": "Size 7 (largest - 1.8em)",
        "input": '<span style="font-size: 1.8em;">huge</span>',
        "expected_contains": ["{\\fs86}huge"]
    },
    {
        "id": "TC24",
        "name": "Multiple sizes in one line",
        "input": '<span style="font-size: 0.6em;">small</span> normal <span style="font-size: 1.8em;">BIG</span>',
        "expected_contains": ["{\\fs29}small", "{\\fs86}BIG"]
    },

    # 7. Color Tests
    {
        "id": "TC25",
        "name": "Red (#ff0000)",
        "input": '<span style="color: #ff0000;">red</span>',
        "expected_contains": ["{\\c&H000000FF}red"]
    },
    {
        "id": "TC26",
        "name": "Green (#00ff00)",
        "input": '<span style="color: #00ff00;">green</span>',
        "expected_contains": ["{\\c&H0000FF00}green"]
    },
    {
        "id": "TC27",
        "name": "Blue (#0000ff)",
        "input": '<span style="color: #0000ff;">blue</span>',
        "expected_contains": ["{\\c&H00FF0000}blue"]
    },
    {
        "id": "TC29",
        "name": "Multiple colors in one line",
        "input": '<span style="color: #ff0000;">red</span>, <span style="color: #00ff00;">green</span>, <span style="color: #0000ff;">blue</span>',
        "expected_contains": ["{\\c&H000000FF}red", "{\\c&H0000FF00}green", "{\\c&H00FF0000}blue"]
    },

    # 8. Special Characters and Edge Cases
    {
        "id": "TC30",
        "name": "Text with apostrophe",
        "input": "<b>it's</b> working",
        "expected_contains": ["{\\b1}it's{\\b0}"]
    },
    {
        "id": "TC31",
        "name": "Text with quotes",
        "input": '<span style="color: #ffff00;">"quoted text"</span>',
        "expected_contains": ["{\\c&H0000FFFF}\"quoted text\""]
    },
    {
        "id": "TC32",
        "name": "Text with ampersand",
        "input": '<b>R&B</b> music',
        "expected_contains": ["{\\b1}R&B{\\b0}"]
    },
    {
        "id": "TC33",
        "name": "Numbers mixed with text",
        "input": '<b>Chapter 1</b>: The Beginning',
        "expected_contains": ["{\\b1}Chapter 1{\\b0}"]
    },

    # 9. RTL Text (Arabic) with Styling
    {
        "id": "TC34",
        "name": "Arabic text with color",
        "input": '<span style="color: #ffff00;">نعم</span>',
        "expected_contains": ["{\\c&H0000FFFF}نعم"]
    },
    {
        "id": "TC35",
        "name": "Mixed English and Arabic with different styles",
        "input": '<b>ابن المسعود</b> <span style="color: #00ffff;">radi allahu anhu</span>',
        "expected_contains": ["{\\b1}المسعود", "{\\b1}ابن", "{\\c&H00FFFF00}radi"]
    },
    {
        "id": "TC36",
        "name": "Arabic with bold + color",
        "input": '<span style="color: #ffff00;"><b>صلى الله عليه وسلم</b></span>',
        "expected_contains": ["{\\c&H0000FFFF}", "{\\b1}وسلم", "{\\b1}عليه", "{\\b1}الله", "{\\b1}صلى"]
    },

    # 13. Complex Real-World Examples
    {
        "id": "TC46",
        "name": "Karaoke-style highlighting",
        "input": '<span style="color: #ffffff;">You are </span><span style="color: #ffff00;"><b>my sunshine</b></span>',
        "expected_contains": ["You are", "{\\c&H0000FFFF}", "{\\b1}my sunshine"]
    },
    {
        "id": "TC47",
        "name": "Speaker emphasis",
        "input": '<span style="color: #00ffff;">Messenger:</span> <b>Peace be upon him</b>',
        "expected_contains": ["{\\c&H00FFFF00}Messenger:", "{\\b1}Peace be upon him"]
    },
    {
        "id": "TC48",
        "name": "Word-by-word emphasis",
        "input": '<b>This</b> is <b>very</b> <b>important</b>',
        "expected_contains": ["{\\b1}This{\\b0}", "{\\b1}very{\\b0}", "{\\b1}important{\\b0}"]
    },

    # 14. Legacy Font Tag Tests
    {
        "id": "TC43",
        "name": "Old font tag with color",
        "input": '<font color="#ffff00">yellow</font>',
        "expected_contains": ["{\\c&H0000FFFF}yellow"]
    },
    {
        "id": "TC44",
        "name": "Old font tag with size",
        "input": '<font size="6">big</font>',
        "expected_contains": ["{\\fs77}big"]
    },
    {
        "id": "TC45",
        "name": "Old font tag with both",
        "input": '<font color="#00ffff" size="7">big cyan</font>',
        "expected_contains": ["{\\c&H00FFFF00}", "{\\fs86}big cyan"]
    },
]


def run_test(test_case):
    """Run a single test case"""
    formatter = ASSFormatter({})
    result = formatter.parse_html_to_ass(test_case["input"])

    passed = True
    missing = []

    for expected in test_case["expected_contains"]:
        if expected not in result:
            passed = False
            missing.append(expected)

    return {
        "passed": passed,
        "result": result,
        "missing": missing
    }


def main():
    """Run all tests and display results"""
    print("=" * 80)
    print("ASS FORMATTER TEST SUITE")
    print("=" * 80)
    print()

    passed_count = 0
    failed_count = 0
    failed_tests = []

    for test_case in test_cases:
        result = run_test(test_case)

        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} | {test_case['id']} | {test_case['name']}")

        if not result["passed"]:
            failed_count += 1
            failed_tests.append(test_case)
            print(f"  Input:    {test_case['input']}")
            print(f"  Output:   {result['result']}")
            print(f"  Missing:  {', '.join(result['missing'])}")
        else:
            passed_count += 1

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total:  {len(test_cases)}")
    print(f"Passed: {passed_count} ✅")
    print(f"Failed: {failed_count} ❌")
    print()

    if failed_tests:
        print("FAILED TESTS:")
        for test in failed_tests:
            print(f"  - {test['id']}: {test['name']}")
        print()

    return failed_count == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
