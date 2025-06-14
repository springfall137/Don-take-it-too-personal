import { connectDB } from "@/util/database.js"
import Link from "next/link"
import DetailLink from "./DetailLink"
import ListItem from "./ListItem"

export const dynamic = 'force-dynamic'

export default async function List() {
    const db = (await connectDB).db("forum")
    let result = await db.collection('post').find().toArray()
    // console.log(result)

    // let data = { name: 'kim', age: 20 }
    // console.log(data.name)

    // console.log(result[0].title)

    return (
        <div className="list-bg">
            {
                <ListItem result={JSON.parse(JSON.stringify(result))} />
            }


            {/* <div className="list-item">
                <h4>{ result[0].title }</h4>
                <p>1월 1일</p>
            </div>
            <div className="list-item">
                <h4>{result[1].title}</h4>
                <p>1월 1일</p>
            </div>
            <div className="list-item">
                <h4>{result[2].title}</h4>
                <p>1월 1일</p>
            </div> */}
        </div>
    )
}
