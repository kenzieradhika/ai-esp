// Optional on-device screen for the demo: the story appears on a display wired
// to the ESP32 itself, no laptop. Two panels supported via DISPLAY_KIND:
//
//   DISPLAY_OLED_I2C (default) -- 1.3" 128x64 I2C mono OLED, SH1106 controller
//     (the common 1.3" panel). 4 wires: GND, VCC->3V3, SCL->GPIO9, SDA->GPIO8.
//     If the image is shifted/garbled it's an SSD1306 instead -> set
//     OLED_CONTROLLER to SSD1306 below.
//   DISPLAY_TFT_SPI -- 2.0" 240x320 SPI ST7789 (GMT020-02-7P). Wiring in the TFT
//     block below. Nicer color hero shot for later.
//
// The API (display_begin / display_puts) is identical for both, so the sketch
// integration never changes -- only this header and the wiring.
#ifndef DISPLAY_H
#define DISPLAY_H

#define DISPLAY_OLED_I2C 1
#define DISPLAY_TFT_SPI  2
#ifndef DISPLAY_KIND
#define DISPLAY_KIND DISPLAY_OLED_I2C
#endif

// ======================= 1.3" I2C OLED (SH1106) =============================
#if DISPLAY_KIND == DISPLAY_OLED_I2C
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Wire.h>

#define OLED_SDA 8
#define OLED_SCL 9
#define OLED_ADDR 0x3C          // some panels are 0x3D
#define SCR_W 128
#define SCR_H 64
#define CW 6                    // 6x8 base glyph at text size 1
#define CH 8

static Adafruit_SH1106G oled = Adafruit_SH1106G(SCR_W, SCR_H, &Wire, -1);
static int ox = 0, oy = 0;

static void display_home() {
  oled.clearDisplay();
  ox = 0; oy = 0;
}

static void display_begin() {
  Wire.begin(OLED_SDA, OLED_SCL);
  oled.begin(OLED_ADDR, true);
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SH110X_WHITE);
  oled.setTextWrap(false);      // we wrap at token boundaries ourselves
  oled.display();
  display_home();
}

// Append a token's bytes; whole-token wrap, clear+home when the screen fills,
// flush the framebuffer once per token so text visibly appears.
static void display_puts(const unsigned char *s, int len) {
  if (ox + len * CW > SCR_W) { oy += CH; ox = 0; }
  if (oy + CH > SCR_H) display_home();
  for (int i = 0; i < len; i++) {
    char c = (char)s[i];
    if (c == '\n') { oy += CH; ox = 0; }
    else if (c >= 32 && c < 127) {
      if (ox + CW > SCR_W) { oy += CH; ox = 0; }
      if (oy + CH > SCR_H) display_home();
      oled.setCursor(ox, oy);
      oled.write(c);
      ox += CW;
    }
    if (oy + CH > SCR_H) display_home();
  }
  oled.display();
}

// ======================= 2.0" SPI TFT (ST7789) ==============================
#else
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
#define CHAR_W (6 * TFT_TEXTSIZE)
#define CHAR_H (8 * TFT_TEXTSIZE)
#define LINE_H (CHAR_H + 2)

static Adafruit_ST7789 tft = Adafruit_ST7789(&SPI, TFT_CS, TFT_DC, TFT_RST);

static void display_home() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(2, 2);
}

static void display_begin() {
  SPI.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);
  tft.init(TFT_W, TFT_H);
  tft.setRotation(0);
  tft.setTextSize(TFT_TEXTSIZE);
  tft.setTextColor(ST77XX_GREEN, ST77XX_BLACK);
  tft.setTextWrap(false);
  display_home();
}

static void display_puts(const unsigned char *s, int len) {
  if (tft.getCursorX() + len * CHAR_W > TFT_W)
    tft.setCursor(2, tft.getCursorY() + LINE_H);
  if (tft.getCursorY() + LINE_H > TFT_H)
    display_home();
  for (int i = 0; i < len; i++) {
    char c = (char)s[i];
    if (c == '\n') {
      tft.setCursor(2, tft.getCursorY() + LINE_H);
    } else if (c >= 32 && c < 127) {
      if (tft.getCursorX() + CHAR_W > TFT_W)
        tft.setCursor(2, tft.getCursorY() + LINE_H);
      tft.write(c);
    }
    if (tft.getCursorY() + LINE_H > TFT_H)
      display_home();
  }
}

#endif
#endif
