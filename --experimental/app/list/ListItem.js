'use client'

import Link from "next/link"

export default function ListItem(props) {

    return (
        <div>
            {
                props.result.map((a, i) =>
                    <div className="list-item" key={i}>
                        <Link href={'detail/' + props.result[i]._id}>
                            <h4>{props.result[i].title}</h4>
                        </Link>
                        <Link href={"/edit/" + props.result[i]._id}>✏️</Link>
                        <span className="clickable" onClick={(e) => {
                            // fetch('/api/test?name=kim&age=20')
                            fetch('/api/post/delete', { method: 'POST', body: props.result[i]._id }).then((r) => { return r.json() }).then(() => {
                                e.target.parentElement.style.opacity = 0;
                                setTimeout(() => {
                                    e.target.parentElement.style.display = 'none'
                                }, 1000)
                            })
                        }}>🗑️</span>
                        <p>1월 1일</p>
                    </div>

                )
            }
        </div>
    )
}
