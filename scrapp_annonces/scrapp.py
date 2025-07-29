import scrapy
import pandas as pd
from scrapy.crawler import CrawlerProcess


class AppartSpider(scrapy.Spider):
    name = 'appartSpider'
    async def start(self):
        urls = ['https://www.seloger.com/classified-search?distributionTypes=Buy,Buy_Auction&estateTypes=House,Apartment&locations=AD08FR31096']
        
        for url in urls:
            print(url)
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/115.0 Safari/537.36"}
            )

    async def parse(self, response):
        titles = response.xpath('//a[@data-testid="card-mfe-covering-link-testid"]/@title').getall()
        adresse = response.xpath("//div[@data-testid='cardmfe-description-box-address']/text()").getall()
        m2 = response.xpath("//div[@data-testid='cardmfe-keyfacts-testid']/div[5]/text()").getall()
        print(titles[0])
        print(adresse[0])
        # print(m2)
        yield

if __name__ == "__main__":
    process = CrawlerProcess(settings={
        # "FEEDS": {"resultats.json": {"format": "json"}},
        "LOG_LEVEL": "ERROR",
    })
    process.crawl(AppartSpider)
    process.start()



