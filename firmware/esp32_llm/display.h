// Optional on-device TFT output for the demo: the story appears on a screen
// attached to the ESP32 itself, no laptop. Target panel: GMT020-02-7P, a 2.0"
// 240x320 ST7789 (standard resolution -> no column/row offset quirk).
//
// Wiring (display -> ESP32-S3, all safe user GPIOs):
//   GND->GND  VCC->3V3  SCL->GPIO12  SDA->GPIO11  RST->GPIO6  DC->GPIO7  CS->GPIO10
//   (backlight is always-on via the module's Q1 transistor; no BL pin)
//
// Rendering: tokens are appended live with token-boundary word-wrap; when the
// text reaches the bottom the screen clears and restarts at the top, so on
// camera the story continuously writes itself.
#ifndef DISPLAY_H
#define DISPLAY_H

#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>

#define TFT_CS 10
#define TFT_DC 7
#define TFT_RST 6
#define TFT_SCK 12
#define TFT_MOSI 11

#define TFT_W 240
#define TFT_H 320
#define TFT_TEXTSIZE 2
#define CHAR_W (6 * TFT_TEXTSIZE)   // Adafruit GFX base glyph is 6x8
#define CHAR_H (8 * TFT_TEXTSIZE)
#define LINE_H (CHAR_H + 2)

static Adafruit_ST7789 tft = Adafruit_ST7789(&SPI, TFT_CS, TFT_DC, TFT_RST);

static void display_home() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(2, 2);
}

static void display_begin() {
  SPI.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);  // write-only, no MISO
  tft.init(TFT_W, TFT_H);                     // 240x320, no offset
  tft.setRotation(0);                         // portrait, story scrolls downward
  tft.setTextSize(TFT_TEXTSIZE);
  tft.setTextColor(ST77XX_GREEN, ST77XX_BLACK);
  tft.setTextWrap(false);                     // we wrap at token boundaries ourselves
  display_home();
}

// Append one token's bytes, wrapping whole tokens and clearing when full.
static void display_puts(const unsigned char *s, int len) {
  // If this token would overrun the line, break to the next line first
  // (keeps words intact instead of splitting mid-token).
  if (tft.getCursorX() + len * CHAR_W > TFT_W)
    tft.setCursor(2, tft.getCursorY() + LINE_H);
  if (tft.getCursorY() + LINE_H > TFT_H)
    display_home();

  for (int i = 0; i < len; i++) {
    char c = (char)s[i];
    if (c == '\n') {
      tft.setCursor(2, tft.getCursorY() + LINE_H);
    } else if (c >= 32 && c < 127) {         // printable ASCII (TinyStories domain)
      if (tft.getCursorX() + CHAR_W > TFT_W)
        tft.setCursor(2, tft.getCursorY() + LINE_H);
      tft.write(c);
    }
    if (tft.getCursorY() + LINE_H > TFT_H)
      display_home();
  }
}

#endif
