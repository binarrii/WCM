const puppeteer = require('puppeteer-core');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: 'new'
  });
  const page = await browser.newPage();
  await page.goto('file:///Users/binarii/workspaces/binarii/WCM/docs/dataset_creation_standards.html', {waitUntil: 'networkidle0'});
  await page.pdf({
    path: 'dataset_creation_standards.pdf',
    format: 'A4',
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size: 10px; width: 100%; text-align: center; color: #555;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: {
      top: '30px',
      bottom: '50px'
    }
  });
  await browser.close();
  console.log("PDF generated successfully.");
})();
