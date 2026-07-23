const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

(async () => {
  const artifactDir = '/Users/binarii/.gemini/antigravity-cli/brain/17dbb2ee-9110-4a81-bad5-2de9004e3ce9';
  
  console.log('Launching browser...');
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: 'new',
    defaultViewport: { width: 1280, height: 800 }
  });
  
  const page = await browser.newPage();
  
  // Listen to page console and error logs
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));
  
  // Navigate to Vite local server
  console.log('Navigating to WebUI...');
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle2' });
  
  // Take initial screenshot
  await page.screenshot({ path: path.join(artifactDir, 'test_initial.png') });
  console.log('Initial dashboard captured.');
  
  // Function to click add button
  const openModal = async () => {
    const addBtn = await page.$('.add-btn');
    await addBtn.click();
    await page.waitForSelector('.modal-card', { visible: true });
  };
  
  // Test Case 1: Upload multi-face image (Should fail)
  console.log('\n--- Test Case 1: Multi-Face Image Upload (Expect Failure) ---');
  await openModal();
  
  await page.type('#name', '多脸测试');
  await page.select('#type', '其它');
  await page.type('#remarks', '系统应自动拦截包含多张人脸的合照。');
  
  // Upload multi-face image
  const multiFacePath = '/Users/binarii/Downloads/2026智能审核/明星艺人/0000000000004664_779c183347a83b8c6aac9ee5cea7543c.png';
  const fileInput = await page.$('.hidden-input');
  await fileInput.uploadFile(multiFacePath);
  
  // Wait for cropper modal and click confirm crop
  await page.waitForSelector('.cropper-confirm-btn', { visible: true });
  await page.screenshot({ path: path.join(artifactDir, 'test_case1_preview.png') });
  await page.click('.cropper-confirm-btn');
  await page.waitForSelector('.cropper-confirm-btn', { hidden: true });
  await new Promise(r => setTimeout(r, 1000)); // Wait for Vue reactive DOM updates to stabilize
  
  // Click submit
  console.log('Submitting multi-face image...');
  const submitBtn = await page.$('button[type="submit"]');
  await submitBtn.click();
  
  // Wait for toast error card
  console.log('Waiting for error validation toast...');
  await page.waitForSelector('.toast-card.error', { visible: true, timeout: 15000 });
  const errorToastText = await page.$eval('.toast-card.error .toast-message', el => el.textContent);
  console.log('Verification Passed. Error message received: ', errorToastText);
  
  await page.screenshot({ path: path.join(artifactDir, 'test_case1_failed.png') });
  
  // Close modal
  const closeBtn = await page.$('.close-btn');
  await closeBtn.click();
  await page.waitForSelector('.modal-card', { hidden: true });
  
  // Test Case 2: Upload single-face image (Should succeed)
  console.log('\n--- Test Case 2: Single-Face Image Upload (Expect Success) ---');
  await openModal();
  
  await page.type('#name', '张铁柱');
  await page.select('#type', '落马官员');
  await page.type('#remarks', '涉嫌严重违纪违法。');
  
  const singleFacePath = '/Users/binarii/Downloads/2026智能审核/劣迹艺人/0000000000016508_164ad27ae0448e990c63dc3e9a2474cc.jpg';
  const fileInput2 = await page.$('.hidden-input');
  await fileInput2.uploadFile(singleFacePath);
  
  // Wait for cropper modal and click confirm crop
  await page.waitForSelector('.cropper-confirm-btn', { visible: true });
  await page.screenshot({ path: path.join(artifactDir, 'test_case2_preview.png') });
  await page.click('.cropper-confirm-btn');
  await page.waitForSelector('.cropper-confirm-btn', { hidden: true });
  await new Promise(r => setTimeout(r, 1000)); // Wait for Vue reactive DOM updates to stabilize
  
  // Click submit
  console.log('Submitting single-face image...');
  const submitBtn2 = await page.$('button[type="submit"]');
  await submitBtn2.click();
  
  console.log('Waiting for success confirmation...');
  try {
    await page.waitForSelector('.toast-card.success', { visible: true, timeout: 15000 });
    const successToastText = await page.$eval('.toast-card.success .toast-message', el => el.textContent);
    console.log('Verification Passed. Success message received: ', successToastText);
    await page.screenshot({ path: path.join(artifactDir, 'test_case2_success.png') });
  } catch (err) {
    console.log('Success toast not found. Checking for error toast...');
    const errorToastExists = await page.waitForSelector('.toast-card.error', { visible: true, timeout: 2000 }).catch(() => null);
    if (errorToastExists) {
      const errorMsg = await page.$eval('.toast-card.error .toast-message', el => el.textContent);
      console.log('Backend returned error: ', errorMsg);
    } else {
      console.log('No error toast found.');
    }
    await page.screenshot({ path: path.join(artifactDir, 'test_case2_failed.png') });
    throw err;
  }
  
  // Wait for card to appear
  console.log('Verifying final record lists...');
  await page.waitForSelector('.record-card', { visible: true });
  await page.screenshot({ path: path.join(artifactDir, 'test_final_dashboard.png') });
  
  // Test Case 3: Test Scroll Loading (Waterfall pagination)
  console.log('\n--- Test Case 3: Test Scroll Loading (Waterfall Pagination) ---');
  const initialCardsCount = await page.$$eval('.record-card', cards => cards.length);
  console.log(`Initial visible cards count: ${initialCardsCount}`);
  
  // Scroll to the bottom of the page
  console.log('Scrolling to bottom of the page...');
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  
  // Wait for new cards to be fetched and loaded
  console.log('Waiting for new records to load...');
  await page.waitForFunction(
    (prevCount) => document.querySelectorAll('.record-card').length > prevCount,
    { timeout: 15000 },
    initialCardsCount
  );
  
  const finalCardsCount = await page.$$eval('.record-card', cards => cards.length);
  console.log(`Waterfall scroll loaded successfully! Visible cards count increased to: ${finalCardsCount}`);
  await page.screenshot({ path: path.join(artifactDir, 'test_case3_scroll_loaded.png') });
  
  // Test Case 4: Image Search (人脸检索)
  console.log('\n--- Test Case 4: Image Search (人脸检索) ---');
  // Click image search button inside the search box
  const searchImgBtn = await page.waitForSelector('.search-image-btn', { visible: true });
  await searchImgBtn.click();
  
  // Wait for the Image Search modal to open
  await page.waitForSelector('.modal-card', { visible: true });
  console.log('Image Search modal opened.');
  
  // Upload search face image (using the singleFacePath file input inside the open modal)
  const searchFileInput = await page.waitForSelector('.modal-overlay .hidden-input');
  await searchFileInput.uploadFile(singleFacePath);
  
  // Wait for the preview image to load
  await page.waitForSelector('.upload-preview', { visible: true });
  await page.screenshot({ path: path.join(artifactDir, 'test_case4_preview.png') });
  
  // Click search button
  console.log('Executing Image Search...');
  const confirmSearchBtn = await page.waitForSelector('.modal-overlay .btn-primary');
  await confirmSearchBtn.click();
  
  // Wait for image search banner to show up
  await page.waitForSelector('.image-search-banner', { visible: true });
  console.log('Image Search active. Results loaded successfully!');
  
  // Verify matching similarity badges exist on cards
  const similarityBadgeExists = await page.waitForSelector('.similarity-badge', { visible: true });
  const similarityVal = await page.$eval('.similarity-badge', el => el.textContent);
  console.log(`Matching record found with similarity rating: ${similarityVal.trim()}`);
  
  await page.screenshot({ path: path.join(artifactDir, 'test_case4_success.png') });
  
  // Clear image search
  console.log('Clearing Image Search...');
  const clearSearchBtn = await page.waitForSelector('.clear-image-search-btn');
  await clearSearchBtn.click();
  
  // Wait for banner to disappear
  await page.waitForSelector('.image-search-banner', { hidden: true });
  console.log('Image Search cleared. Returned to paginated records.');
  
  await browser.close();
  console.log('\nAutomation tests completed successfully.');
})();
