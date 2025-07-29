import scrapy
import pandas as pd
from scrapy.crawler import CrawlerProcess


class AppartSpider(scrapy.Spider):
    name = 'appartSpider'
    async def start(self):
        urls = []
        
        for url in urls:
            print(url)
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/115.0 Safari/537.36"}
            )

    async def parse(self, response):
        titles = response.xpath('').getall()
        print(titles[0])
        yield

if __name__ == "__main__":
    process = CrawlerProcess(settings={
        # "FEEDS": {"resultats.json": {"format": "json"}},
        "LOG_LEVEL": "ERROR",
    })
    process.crawl(AppartSpider)
    process.start()



